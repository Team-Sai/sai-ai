"""
대화형 특약에서 사용자 답장을 규칙으로 분류한다. (LLM 0번)

확실할 때만 분류하고, 애매하면 AMBIGUOUS로 돌려준다.
- 버튼(선택지)과 같은 답이 가장 확실하다 → 먼저 본다
- 값이 함께 있으면 값이 이긴다 ("아니, 2개월로" → 값 바꾸기)
- "네/아니요"는 그 한 단어(+문장부호)일 때만 확정/거절
- 알아들은 내용을 되풀이한다 (echo)
"""

from __future__ import annotations

import copy
import re
from typing import Any

from sai_rag.agent.slot_rules import (
    contract_value_text,
    ro_particle,
    tagged_values,
)


CONFIRM_WORDS = {"네", "예", "응", "맞아요", "맞아", "좋아요", "좋아", "그래요", "그래", "ㅇㅇ", "작성해주세요"}
REJECT_WORDS = {"아니요", "아니오", "아니", "아뇨", "아니에요", "틀려요"}

# 띄어쓰기를 지운 글에서 찾는다.
UNKNOWN_PHRASES = ("몰라", "모르겠", "나중에", "아직안정했", "미정")

# "2개월 이상이면"처럼 조건을 바꾸는 말은 값 바꾸기로 보지 않는다.
CONDITION_WORDS = ("이상", "이하", "초과", "미만", "이면", "부터", "까지", "경우", "때")

NATIVE_NUMBERS = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}
NATIVE_PERIOD_PATTERN = re.compile(r"(다섯|여섯|일곱|여덟|아홉|한|두|세|네|열)\s*(?:달|개월)")
FROM_TO_PATTERN = re.compile(r"에서\s*(.+?)\s*(?:으로|로)")
IGNORED_CHARS = re.compile(r"[\s.,!?~ㅎㅋ]+")

PERIOD_KINDS = {"개월": "MONTHS", "일": "DAYS", "회": "COUNT"}

UNKNOWN_ANSWER = "UNKNOWN"


def _normalize(text: str) -> str:
    return IGNORED_CHARS.sub("", text)


def _reply_values(message: str) -> set[tuple[str, Any]]:
    """답장의 값을 (값 칸 종류, 값)으로 꺼낸다. 예: "네 달로" → {("MONTHS", 4)}"""

    text = NATIVE_PERIOD_PATTERN.sub(
        lambda match: f"{NATIVE_NUMBERS[match.group(1)]}개월",
        message,
    )

    from_to = FROM_TO_PATTERN.search(text)

    if from_to:
        # "5%에서 7%로"는 바꿀 값(7%)만 본다.
        text = from_to.group(1)

    values: set[tuple[str, Any]] = set()

    for tag, value in tagged_values(text):
        if tag == "PERIOD":
            number, unit = value
            values.add((PERIOD_KINDS.get(unit, "OTHER"), number))
        else:
            values.add((tag, value))

    return values


def _value_text(kind: str, value: Any) -> str:
    if kind == "PERCENT":
        return f"{value:g}%"

    if kind == "AMOUNT":
        return f"{value:,}원"

    if kind == "DATE":
        return f"{value.year}년 {value.month}월 {value.day}일"

    if kind == "MONTHS":
        return f"{value}개월"

    if kind == "DAYS":
        return f"{value}일"

    return f"{value}회"


def _match_option(text: str, question: dict[str, Any]) -> str | None:
    """답장이 질문의 선택지 중 하나와 같으면 그 선택지. "네"는 "예"·"맞아요"와 같은 것으로 본다."""

    for option in question.get("options") or []:
        normalized = _normalize(option)

        if text == normalized:
            return option

        if text in CONFIRM_WORDS and normalized in CONFIRM_WORDS:
            return option

        if text in REJECT_WORDS and normalized in REJECT_WORDS:
            return option

    return None

def _find_answered_question(
    text: str,
    pending: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    답장이 어느 질문의 답인지 찾는다.
    - 버튼 글자와 똑같으면 질문이 여러 개여도 그 질문 (단, 한 질문에만 있는 버튼일 때)
    - "네/아니요" 같은 짧은 말은 질문이 하나일 때만
    """

    exact = [
        (question, option)
        for question in pending
        for option in question.get("options") or []
        if _normalize(option) == text
    ]

    if len(exact) == 1:
        return exact[0]

    if not exact and len(pending) == 1:
        option = _match_option(text, pending[0])

        if option is not None:
            return pending[0], option

    return None, None


def route_reply(
    *,
    message: str,
    contract: dict[str, Any],
    slots: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    answers: dict[str, str],
) -> dict[str, Any]:
    # 받은 값은 건드리지 않고 복사본을 고쳐서 돌려준다.
    slots = copy.deepcopy(slots)
    answers = dict(answers)

    pending = [question for question in questions if question["id"] not in answers]
    text = _normalize(message)

    def done(kind: str, echo: str | None = None) -> dict[str, Any]:
        return {
            "kind": kind,
            "slots": slots,
            "answers": answers,
            "pendingQuestions": [
                question for question in questions if question["id"] not in answers
            ],
            "echo": echo,
            "readyToWrite": kind == "CONFIRM",
        }

    if not text:
        return done("AMBIGUOUS")

    # 1. 선택지(버튼)와 같은 답
    question, option = _find_answered_question(text, pending)

    if question is not None:
        answers[question["id"]] = option

        if question["id"].startswith("confirm_from_"):
            field = question["id"].removeprefix("confirm_from_")

            if _normalize(option) not in CONFIRM_WORDS:
                return done("REJECT", "어떤 값을 어떻게 바꾸고 싶은지 다시 말씀해 주세요.")

            current = contract_value_text(contract, field)

            for slot in slots:
                if slot.get("field") == field:
                    slot["from"] = current

            return done("ANSWER", f"지금 계약 값 {current}에서 바꾸는 걸로 할게요.")

        return done("ANSWER", f"{option}{ro_particle(option)} 정했어요.")

    # 1-1. 버튼이 없는 질문(이름·금액처럼 직접 적는 질문)은 적은 말을 그대로 그 질문의 답으로 쓴다.
    #      화면은 남은 질문 중 첫 번째만 보여 주므로 대상은 분명하다.
    current = pending[0] if pending else None

    if (
        current is not None
        and not current.get("options")
        and not any(phrase in text for phrase in UNKNOWN_PHRASES)
        and text not in CONFIRM_WORDS
        and text not in REJECT_WORDS
    ):
        answer = message.strip()
        answers[current["id"]] = answer

        return done("ANSWER", f"{answer}{ro_particle(answer)} 정했어요.")


    # 2. 값이 함께 있으면 값이 이긴다
    values = _reply_values(message)

    if values:
        if len(values) != 1 or any(word in message for word in CONDITION_WORDS):
            return done("AMBIGUOUS")

        kind, value = next(iter(values))
        targets = [slot for slot in slots if slot.get("kind") == kind]

        if len(targets) != 1:
            return done("AMBIGUOUS")

        new_value = _value_text(kind, value)
        targets[0]["value"] = new_value
        targets[0]["confirmed"] = True

        return done(
            "VALUE_CHANGE",
            f"'{targets[0]['name']}' 값을 {new_value}{ro_particle(new_value)} 바꿨어요.",
        )

    # 3. 모름·나중에 → [입력 필요] 칸으로 남긴다
    if any(phrase in text for phrase in UNKNOWN_PHRASES):
        # 화면은 남은 질문 중 첫 번째만 보여 주므로, "나중에"는 그 질문에 대한 답이다.
        if not pending:
            return done("AMBIGUOUS")

        answers[pending[0]["id"]] = UNKNOWN_ANSWER

        return done("UNKNOWN", "그 부분은 계약서에 [입력 필요] 칸으로 남겨 둘게요.")

    # 4. 한 단어 확정·거절
    if text in CONFIRM_WORDS:
        if pending:
            # 질문이 남았는데 "네"만 오면 무엇에 대한 답인지 알 수 없다.
            return done("AMBIGUOUS")

        for slot in slots:
            slot["confirmed"] = True

        return done("CONFIRM", "확인했어요. 이대로 특약을 쓸게요.")

    if text in REJECT_WORDS:
        return done("REJECT", "어떤 부분이 다른지 말씀해 주세요.")

    return done("AMBIGUOUS")