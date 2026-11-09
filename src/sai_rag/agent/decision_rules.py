"""
대화형 특약에서 "정해야 할 것"을 어떻게 다룰지 정한 표.

생성기의 미결정 조건 후보(ClauseGenerator.ISSUE_DECISIONS)를 셋으로 나눈다.
- CONTRACT: 계약 정보에 이미 있는 값 → 계약 값을 그대로 쓴다
- DEFAULT:  보통 정해진 값이 있는 것 → 따로 말이 없으면 기본 문구로 쓴다 (이해 확인 때 보여 준다)
- QUESTION: 사용자가 정해야 하는 것 → 이해 확인 때 규칙으로 묻는다
- REQUEST:  사용자 요청에서 정해지는 것 → 요청에 없으면 지금처럼 [입력 필요]

문구·분류를 바꿀 때는 이 파일만 고친다. 대화형 작성에만 적용한다.
policy=True는 우리 서비스가 정한 값이라 계약 담당 팀원과 맞춰야 한다.
legal_ref가 있고 verified=False면 조문 확인이 아직 필요하다.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

from sai_rag.agent.intent_value_normalizer import find_amounts, find_percents


CONTRACT = "CONTRACT"
DEFAULT = "DEFAULT"
QUESTION = "QUESTION"
REQUEST = "REQUEST"
DIRECT_INPUT_ANSWER = "직접 입력할게요"  # 사용자가 값을 직접 쓰겠다는 답
CLAUSE_SPLIT_PATTERN = re.compile(r"[,.\n]|(?:이고|이며|인데)\s+")  # 문장을 구간으로 나눈다

# 이해 확인 때 기본값을 한 번에 보여 주는 질문. 생성기(A-3)도 이 이름으로 답을 찾는다.
DEFAULTS_QUESTION_ID = "confirm_defaults"
DEFAULTS_KEEP = "좋아요"
DEFAULTS_CHANGE = "바꿀래요"


@dataclass(frozen=True)
class Decision:
    name: str
    kind: str
    # CONTRACT: 어느 계약 값을 쓰는지 / DEFAULT: 기본 문구 / QUESTION: 답이 없을 때 쓸 기본값 ("" = 기본값 없음)
    text: str = ""
    question_id: str = ""
    question: str = ""
    options: tuple[str, ...] = ()
    # 사용자 요청이나 LLM 질문에 이 말이 있으면 "이미 다룬 것"으로 보고 규칙 질문을 더하지 않는다.
    keywords: tuple[str, ...] = ()
    policy: bool = False
    legal_ref: str = ""
    verified: bool = False
    value_name: str = ""  # 금액이 없으면 이 이름으로 빈칸을 남긴다
    value_keyword: str = ""  # 이 말이 있는 구간의 금액만 인정한다

    @property
    def has_default(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Notice:
    """쟁점이 있으면 이해 확인과 작성 결과에 항상 붙이는 안내."""

    text: str
    legal_ref: str
    verified: bool


PARTIAL_PAYMENT = Decision(
    name="일부 지급 시 연체 처리 기준",
    kind=QUESTION,
    text="그 회차 상환금을 전혀 지급하지 않은 경우에만 연체로 본다",
    question_id="partial_payment",
    question="일부만 낸 회차도 연체로 볼까요?",
    options=("일부만 내도 연체로 봐요", "전혀 안 낸 회차만 연체로 봐요"),
    keywords=("일부",),
)

NOTICE_METHOD = Decision(
    name="사전 통지 방법",
    kind=DEFAULT,
    text="서면 또는 전자적 방법으로 알린다",
    policy=True,
)


DECISIONS: dict[str, tuple[Decision, ...]] = {
    "DEFAULT": (
        Decision(name="연체 판단 기준", kind=DEFAULT,
                 text="각 회차 상환기일까지 그 회차 상환금을 모두 지급하지 않은 경우 연체로 본다"),
        Decision(name="연체 횟수 산정 기준", kind=DEFAULT, text="연체 횟수는 회차별로 센다"),
        PARTIAL_PAYMENT,
    ),
    "ACCELERATION": (
        Decision(name="기한의 이익 상실 또는 일시변제 발생 요건", kind=REQUEST),
        Decision(name="연속 연체 판단 기준", kind=DEFAULT, text="연이은 회차가 모두 연체된 경우를 연속 연체로 본다"),
        PARTIAL_PAYMENT,
        Decision(
            name="사전 통지 여부",
            kind=QUESTION,
            text="채권자가 미리 알리고 7일이 지난 뒤 기한의 이익을 잃는다",
            question_id="acceleration_notice",
            question="남은 돈을 한꺼번에 갚게 하기 전에 미리 알려 줄까요?",
            options=("미리 알리고 7일 뒤에", "알리지 않고 바로"),
            keywords=("통지", "알리", "알려"),
            policy=True,
        ),
        NOTICE_METHOD,
        Decision(name="변제 유예기간", kind=DEFAULT, text="미리 알린 경우 알린 날부터 7일로 한다", policy=True),
    ),
    "LATE_INTEREST": (
        Decision(name="연체이자율", kind=CONTRACT, text="계약의 연체이율 또는 사용자가 말한 연체이율"),
        Decision(name="연체이자 계산 대상", kind=DEFAULT, text="연체이자는 연체된 금액에 대해 계산한다"),
        Decision(name="연체이자 계산 시작일", kind=DEFAULT, text="연체이자는 상환기일 다음 날부터 계산한다"),
        Decision(name="연체이자 계산 종료일", kind=DEFAULT, text="연체이자는 실제로 모두 갚은 날까지 계산한다"),
        Decision(
            name="연체이자 상한",
            kind=DEFAULT,
            text="연체이자율은 연 20%를 넘지 않는다",
            policy=True,
            legal_ref="이자제한법 제2조제1항, 이자제한법 제2조제1항의 최고이자율에 관한 규정 (연체이자 적용 여부 확인 필요)",
            verified=False,
        ),
    ),
    "INTEREST": (
        Decision(name="약정이율", kind=CONTRACT, text="계약의 약정이율"),
        Decision(name="이자 계산 대상", kind=DEFAULT, text="이자는 남은 원금에 대해 계산한다"),
        Decision(name="이자 계산 방식", kind=DEFAULT, text="1년을 365일로 보고 날짜만큼 계산한다", policy=True),
        Decision(name="이자 지급일", kind=CONTRACT, text="계약의 정기 상환일"),
    ),
    "REPAYMENT": (
        Decision(name="상환 방식", kind=CONTRACT, text="계약의 상환방식"),
        Decision(name="상환일", kind=CONTRACT, text="계약의 정기 상환일"),
        Decision(name="회차별 상환금액", kind=DEFAULT, text="회차별 상환금액은 계약의 상환방식에 따라 계산한다"),
    ),
    "EARLY_REPAYMENT": (
        Decision(
            name="중도상환 가능 범위",
            kind=QUESTION,
            text="중도상환은 남은 원금 전액을 갚는 경우에만 할 수 있다",
            question_id="early_repayment_scope",
            question="일부 금액만 먼저 갚는 것도 허용할까요?",
            options=("전액만 허용", "일부도 허용"),
            keywords=("일부", "전액"),
        ),
        Decision(name="최소 중도상환금액", kind=DEFAULT, text="최소 중도상환금액은 따로 두지 않는다", policy=True),
        Decision(name="사전 통지 여부 및 기간", kind=DEFAULT, text="중도상환하기 전에 채권자에게 알린다", policy=True),
        Decision(name="통지 방법", kind=DEFAULT, text="서면 또는 전자적 방법으로 알린다", policy=True),
        Decision(name="일부 상환 후 상환계획 조정 방식", kind=DEFAULT,
                 text="일부 상환 후에는 남은 원금을 기준으로 이후 회차 금액을 다시 계산한다"),
        Decision(name="중도상환일까지의 이자 정산 방식", kind=DEFAULT, text="이자는 중도상환일까지 날짜만큼 계산한다"),
        Decision(
            name="중도상환 수수료 또는 손해배상 처리",
            kind=QUESTION,
            text="중도상환 수수료는 받지 않는다",
            question_id="early_repayment_fee",
            question="중도상환 수수료를 받을까요?",
            options=("받지 않아요", "받아요"),
            keywords=("수수료", "손해"),
            legal_ref="민법 제468조 (변제기 전 변제 시 상대방의 손해 배상, 해석 확인 필요)",
            verified=False,
        ),
    ),
    "PENALTY": (
        Decision(name="위약금 산정 기준", kind=REQUEST, value_name="위약금 금액", value_keyword="위약금"),
        Decision(name="위약금 적용 대상 금액", kind=REQUEST),
        Decision(
            name="위약금 발생 횟수",
            kind=QUESTION,
            text="위약금은 한 번만 받는다",
            question_id="penalty_count",
            question="위약금은 한 번만 받을까요, 어길 때마다 받을까요?",
            options=("한 번만", "어길 때마다"),
            keywords=("마다", "매번", "횟수"),
        ),
        Decision(name="위약금 산정 기간", kind=REQUEST),
        Decision(
            name="위약금과 지연손해금의 관계",
            kind=QUESTION,
            text="",
            question_id="penalty_with_late_interest",
            question="위약금과 연체이자를 둘 다 받을까요?",
            options=("위약금만 받아요", "둘 다 받아요"),
            keywords=("지연손해금", "연체이자", "둘 다"),
            legal_ref="민법 제398조제4항 (위약금의 약정은 손해배상액의 예정으로 추정)",
            verified=True,
        ),
    ),
    "GUARANTEE": (
        Decision(name="보증 범위", kind=DEFAULT, text="보증은 이 계약의 원금·이자·지연손해금 전부에 대해 한다"),
        Decision(
            name="보증 한도",
            kind=QUESTION,
            text="",
            question_id="guarantee_limit",
            question="보증인이 책임질 최고 금액을 정해 주세요.",
            options=("원금만큼", "직접 입력할게요"),
            keywords=("한도", "최고액", "까지만"),
            legal_ref="보증인 보호를 위한 특별법 제4조 (보증채무의 최고액을 서면으로 특정)",
            verified=True,
        ),
        Decision(name="보증 기간", kind=DEFAULT, text="보증 기간은 이 계약의 채무를 모두 갚을 때까지로 한다"),
    ),
    "COLLATERAL": (
        Decision(name="담보 대상", kind=REQUEST),
        Decision(name="담보 범위", kind=DEFAULT, text="담보는 이 계약에 따른 채무 전부를 위해 제공한다"),
        Decision(name="담보 제공 시점", kind=DEFAULT, text="담보는 계약 체결일에 제공한다"),
    ),
}


NOTICES: dict[str, tuple[Notice, ...]] = {
    "GUARANTEE": (
        Notice(
            text="보증은 보증인이 기명날인 또는 서명한 서면으로 해야 효력이 있고, 전자적 형태로 한 보증은 효력이 없어요.",
            legal_ref="민법 제428조의2제1항",
            verified=True,
        ),
        Notice(
            text="보증인이 책임질 최고 금액(보증채무 최고액)을 서면으로 정해야 해요.",
            legal_ref="보증인 보호를 위한 특별법 제4조",
            verified=True,
        ),
    ),
}


def decisions_for(issue_types: list[str]) -> list[Decision]:
    """쟁점들의 정해야 할 것. 같은 이름이 여러 쟁점에 있으면 한 번만 (먼저 나온 쪽)."""

    result: list[Decision] = []
    seen: set[str] = set()

    for issue_type in issue_types:
        for decision in DECISIONS.get(issue_type, ()):
            if decision.name not in seen:
                seen.add(decision.name)
                result.append(decision)

    return result


def questions_for(issue_types: list[str]) -> list[Decision]:
    """규칙으로 물을 질문. 기본값 없는 질문이 앞, 기본값 있는 질문이 뒤 (상한을 넘으면 뒤에서부터 빠진다)."""

    questions = [d for d in decisions_for(issue_types) if d.kind == QUESTION]
    return [q for q in questions if not q.has_default] + [q for q in questions if q.has_default]


def defaults_for(issue_types: list[str]) -> list[Decision]:
    return [d for d in decisions_for(issue_types) if d.kind == DEFAULT]


def notices_for(issue_types: list[str]) -> list[Notice]:
    result: list[Notice] = []

    for issue_type in issue_types:
        for notice in NOTICES.get(issue_type, ()):
            if notice not in result:
                result.append(notice)

    return result


def policy_values() -> list[Decision]:
    """계약 담당 팀원과 맞출 정책값 목록."""

    return [d for decisions in DECISIONS.values() for d in decisions if d.policy and d.text]


def unverified_legal_refs() -> list[str]:
    """조문 확인이 아직 필요한 항목."""

    refs = [d.legal_ref for ds in DECISIONS.values() for d in ds if d.legal_ref and not d.verified]
    refs += [n.legal_ref for ns in NOTICES.values() for n in ns if not n.verified]
    return list(dict.fromkeys(refs))


def _answer_for(
    decision: Decision,
    message: str,
    questions: list[dict],
    answers: dict[str, str],
) -> str | None:
    """이 항목을 사용자가 이미 정했는지. 답(또는 "요청에 있음")이면 그 값, 아니면 None."""

    if decision.question_id in answers:
        return answers[decision.question_id]

    # LLM이 비슷한 질문을 해서 규칙 질문을 안 더한 경우: 그 질문의 답을 쓴다.
    for question in questions:
        text = question.get("text") or ""

        if question.get("id") in answers and any(keyword in text for keyword in decision.keywords):
            return answers[question["id"]]

    if any(keyword in message for keyword in decision.keywords):
        return "요청에 있음"

    return None


def _has_value(text: str) -> bool:
    return bool(find_amounts(text) or find_percents(text))


def _mentions_value(
    keyword: str,
    message: str,
    questions: list[dict],
    answers: dict[str, str],
) -> bool:
    """요청이나 이 항목에 대한 답에 그 항목의 금액이 있는지 본다."""

    texts = [message, *answers.values()]

    for text in texts:
        for part in CLAUSE_SPLIT_PATTERN.split(text):
            if keyword in part and _has_value(part):
                return True

    # "얼마로 할까요?"에 "30만원"처럼 숫자만 답한 경우
    for question in questions:
        answer = answers.get(question.get("id"))

        if answer and keyword in (question.get("text") or "") and _has_value(answer):
            return True

    return False




def writing_guide(
    *,
    issue_types: list[str],
    message: str,
    questions: list[dict],
    answers: dict[str, str],
    unknown_answer: str,
) -> dict[str, list[str]]:
    """
    대화형 작성 때 생성기에 알려 줄 세 묶음.
    - contract: 계약 정보에서 그대로 쓸 값
    - request:  사용자 요청·확정 조건에서 가져올 것 ([입력 필요]로 남기지 않음)
    - defaults: 따로 정하지 않았으므로 이대로 쓸 기본 문구 ([입력 필요]로 남기지 않음)
    - open:     아직 정하지 않은 것 (이것만 [입력 필요] 후보)
    """

    defaults_rejected = answers.get(DEFAULTS_QUESTION_ID) == DEFAULTS_CHANGE
    guide: dict[str, list[str]] = {"contract": [], "request": [], "defaults": [], "open": []}

    for decision in decisions_for(issue_types):
        if decision.kind == CONTRACT:
            guide["contract"].append(f"{decision.name}: {decision.text}")
            continue

        if decision.kind == REQUEST:
            # 금액이 필요한 항목인데 요청과 답 어디에도 금액이 없으면 빈칸으로 남긴다.
            if decision.value_name and not _mentions_value(
                decision.value_keyword, message, questions, answers
            ):
                guide["open"].append(decision.value_name)
            else:
                guide["request"].append(decision.name)
            continue

        if decision.kind == DEFAULT:
            guide["open" if defaults_rejected else "defaults"].append(
                decision.name if defaults_rejected else decision.text
            )
            continue

        # QUESTION
        answer = _answer_for(decision, message, questions, answers)

        if answer in (unknown_answer, DIRECT_INPUT_ANSWER):            guide["open"].append(decision.name)
        elif answer is not None:
            continue  # 확정 조건(답)이나 요청으로 이미 정해졌다.
        elif decision.has_default and not defaults_rejected:
            guide["defaults"].append(decision.text)
        else:
            guide["open"].append(decision.name)

    return guide

