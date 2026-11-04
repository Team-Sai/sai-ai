"""
이해 확인(대화형 특약 1턴)에서 값 칸(slot)에 적용하는 규칙.

LLM이 뽑은 값을 그대로 믿지 않고, 확실하게 판단할 수 있는 것만 규칙으로 확인한다.
- 원문 확인: 값이 사용자 원문에 실제로 있는지 (inMessage)
- 기존 값 확인: 사용자가 말한 원래 값(from)이 현재 계약 값과 다르면 질문
- 본문 칸 안내: 조건 없이 계약서 칸을 바꾸는 요청이면 특약 대신 안내

판단할 수 없으면 아무것도 하지 않는다(안전한 쪽).
"""

from __future__ import annotations

import re
from typing import Any

from sai_rag.agent.decision_rules import (
    DEFAULTS_CHANGE,
    DEFAULTS_KEEP,
    DEFAULTS_QUESTION_ID,
    Decision,
    defaults_for,
    notices_for,
    questions_for,
)

from sai_rag.agent.contract_context_formatter import ContractContextFormatter
from sai_rag.agent.intent_value_normalizer import (
    find_amounts,
    find_dates,
    find_percents,
)


# 모두 받침으로 끝나서 질문 문장에서 조사는 항상 "은"을 쓴다.
FIELD_LABELS = {
    "principalAmount": "원금",
    "interestRate": "이자율",
    "lateInterestRate": "연체이율",
    "startDate": "시작일",
    "maturityDate": "만기일",
    "repaymentDay": "상환일",
    "repaymentType": "상환방식",
}

# 계약서 화면에 칸이 있는 항목. 조건 없이 바꾸면 특약 대신 이 칸을 고치라고 안내한다.
# 연체이율은 계약서 화면에 칸이 없어서 특약으로만 정할 수 있으므로 뺀다.
FORM_FIELDS = {
    "principalAmount",
    "interestRate",
    "startDate",
    "maturityDate",
    "repaymentDay",
    "repaymentType",
}

# 같은 계약 값이 API 이름과 에이전트 이름으로 섞여 들어올 수 있다.
CONTRACT_KEYS = {
    "principalAmount": ("principalAmount", "principal"),
    "maturityDate": ("maturityDate", "repaymentDate"),
}

PERIOD_PATTERN = re.compile(r"(\d+)\s*(개월|달|일|회|번)")

UNIT_ALIASES = {
    "달": "개월",
    "번": "회",
}

FIELD_EDIT_GUIDE = (
    "계약서의 {labels} 칸을 바꾸면 돼요. "
    "특약으로 넣으면 본문과 부딪혀요."
)


# ======================================
# 값 읽기
# ======================================

def _parse_periods(text: str) -> set[tuple[int, str]]:
    return {
        (int(number), UNIT_ALIASES.get(unit, unit))
        for number, unit in PERIOD_PATTERN.findall(text)
    }


def _values(text: str, kind: str) -> set[Any]:
    """값 종류(kind)에 맞게 글에서 값을 꺼낸다. 못 읽으면 빈 집합."""

    if kind == "PERCENT":
        return find_percents(text)

    if kind == "AMOUNT":
        return find_amounts(text)

    if kind == "DATE":
        return find_dates(text)

    if kind in ("MONTHS", "DAYS", "COUNT"):
        return _parse_periods(text)

    return set()


def is_in_message(text: str, kind: str, message: str) -> bool:
    wanted = _values(text, kind)
    return bool(wanted) and wanted <= _values(message, kind)


# ======================================
# 계약 값
# ======================================

def _contract_value(contract: dict[str, Any], field: str) -> Any:
    for key in CONTRACT_KEYS.get(field, (field,)):
        value = contract.get(key)

        if value not in (None, ""):
            return value

    return None


def _repayment_label(value: Any) -> str | None:
    return ContractContextFormatter.REPAYMENT_TYPE_LABELS.get(str(value))


def matches_contract(text: str, field: str, contract: dict[str, Any]) -> bool | None:
    """
    글(text)의 값이 현재 계약 값과 같은지.
    True 같음 / False 다름 / None 판단할 수 없음
    """

    current = _contract_value(contract, field)

    if current is None:
        return None

    try:
        if field in ("interestRate", "lateInterestRate"):
            wanted = find_percents(text)
            return wanted == {round(float(current), 4)} if wanted else None

        if field == "principalAmount":
            wanted = find_amounts(text)
            return wanted == {int(float(current))} if wanted else None

        if field in ("startDate", "maturityDate"):
            wanted = find_dates(text)
            current_dates = find_dates(str(current))

            if not wanted or not current_dates:
                return None

            return wanted == current_dates

        if field == "repaymentDay":
            wanted = {number for number, unit in _parse_periods(text) if unit == "일"}
            return wanted == {int(current)} if wanted else None

        if field == "repaymentType":
            label = _repayment_label(current)

            if label is None:
                return None

            return label.replace("상환", "") in text.replace(" ", "")

    except (TypeError, ValueError):
        return None

    return None


def contract_value_text(contract: dict[str, Any], field: str) -> str | None:
    """질문에 보여 줄 현재 계약 값. 예: 5%, 1,000,000원, 2027년 1월 31일"""

    current = _contract_value(contract, field)

    if current is None:
        return None

    try:
        if field in ("interestRate", "lateInterestRate"):
            return f"{float(current):g}%"

        if field == "principalAmount":
            return f"{int(float(current)):,}원"

        if field in ("startDate", "maturityDate"):
            dates = find_dates(str(current))

            if not dates:
                return None

            day = next(iter(dates))
            return f"{day.year}년 {day.month}월 {day.day}일"

        if field == "repaymentDay":
            return f"매월 {int(current)}일"

        if field == "repaymentType":
            return _repayment_label(current)

    except (TypeError, ValueError):
        return None

    return None


# ======================================
# 조사 (이에요/예요, 으로/로)
# ======================================

def _final_consonant(word: str) -> int | None:
    """마지막 글자의 받침 번호. 받침 없으면 0, 한글이 아니면 None."""

    last = word.strip()[-1:]

    if not last or not ("가" <= last <= "힣"):
        return None

    return (ord(last) - ord("가")) % 28


def _ieyo(word: str) -> str:
    # "%"처럼 한글이 아니면 받침 없는 쪽("퍼센트"로 읽음)
    return "이에요" if _final_consonant(word) else "예요"


def ro_particle(word: str) -> str:
    # 받침이 없거나 ㄹ 받침(8번)이면 "로", 그 밖의 받침이면 "으로"
    return "로" if _final_consonant(word) in (None, 0, 8) else "으로"


# ======================================
# 규칙
# ======================================

def build_from_question(slot: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any] | None:
    field = slot.get("field")
    from_text = slot.get("from")

    if not field or not from_text:
        return None

    if matches_contract(from_text, field, contract) is not False:
        return None

    current = contract_value_text(contract, field)

    if current is None:
        return None

    value = slot["value"]
    label = FIELD_LABELS[field]

    return {
        "id": f"confirm_from_{field}",
        "text": (
            f"지금 계약 {label}은 {current}{_ieyo(current)}. "
            f"{current}에서 {value}{ro_particle(value)} 바꾸는 게 맞을까요?"
        ),
        "options": ["맞아요", "아니요"],
    }


def find_field_edits(slots: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []

    for slot in slots:
        field = slot.get("field")

        if field not in FORM_FIELDS or slot.get("condition"):
            continue

        # 지금 계약 값과 같은 값이면 바꾸는 요청이 아니다.
        if matches_contract(slot["value"], field, contract) is True:
            continue

        edits.append(
            {
                "field": field,
                "label": FIELD_LABELS[field],
                "value": slot["value"],
                "current": contract_value_text(contract, field),
            }
        )

    return edits

def _as_question(decision: Decision) -> dict[str, Any]:
    return {
        "id": decision.question_id,
        "text": decision.question,
        "options": list(decision.options),
    }


def build_defaults_question(texts: list[str]) -> dict[str, Any]:
    """기본값을 몰래 쓰지 않고, 이해 확인 때 한 번에 보여 주고 확인받는다."""

    lines = "\n".join(f"· {text}" for text in texts)

    return {
        "id": DEFAULTS_QUESTION_ID,
        "text": f"따로 말씀이 없으면 이렇게 넣을게요:\n{lines}",
        "options": [DEFAULTS_KEEP, DEFAULTS_CHANGE],
    }




def apply_slot_rules(
    *,
    contract: dict[str, Any],
    message: str,
    understanding: dict[str, Any],
    questions: list[dict[str, Any]],
    max_questions: int,
    issue_types: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    issue_types = list(issue_types)
    slots = understanding["slots"]

    for slot in slots:
        slot["inMessage"] = is_in_message(slot["value"], slot["kind"], message) and (
            not slot.get("from") or is_in_message(slot["from"], slot["kind"], message)
        )

    field_edits = find_field_edits(slots, contract)

    if field_edits:
        labels = ", ".join(edit["label"] for edit in field_edits)

        return {
            "understanding": understanding,
            "questions": [],
            "fieldEdits": field_edits,
            "guide": FIELD_EDIT_GUIDE.format(labels=labels),
            "notices": [],
        }

    # 1. 기존 값 확인 질문 (계약 값과 다른 "원래 값")
    from_questions: list[dict[str, Any]] = []

    for slot in slots:
        question = build_from_question(slot, contract)

        if question and question["id"] not in {q["id"] for q in from_questions}:
            from_questions.append(question)

    used_ids = {question["id"] for question in from_questions}
    llm_questions = [question for question in questions if question["id"] not in used_ids]
    llm_text = " ".join(question.get("text") or "" for question in llm_questions)
    llm_ids = {question["id"] for question in llm_questions}

    # 2. 쟁점별 필수 질문. 사용자가 이미 말했거나 LLM이 비슷한 걸 물었으면 더하지 않는다.
    required = [
        decision
        for decision in questions_for(issue_types)
        if decision.question_id not in llm_ids
        and not any(keyword in message or keyword in llm_text for keyword in decision.keywords)
    ]
    no_default = [_as_question(d) for d in required if not d.has_default]
    with_default = [_as_question(d) for d in required if d.has_default]

    # 3. 기존 값 확인 → 기본값 없는 필수 → LLM → 기본값 있는 필수 순서로, 앞에서부터 상한까지
    kept = (from_questions + no_default + llm_questions + with_default)[:max_questions]
    kept_ids = {question["id"] for question in kept}

    # 4. 기본값: 표의 기본값 + 상한에 밀려 묻지 못한 질문의 기본값
    default_texts = [d.text for d in defaults_for(issue_types)]
    default_texts += [
        d.text for d in required if d.has_default and d.question_id not in kept_ids
    ]

    if default_texts:
        # 요약 확인이라 상한에 세지 않고 맨 끝에 둔다.
        kept.append(build_defaults_question(default_texts))

    return {
        "understanding": understanding,
        "questions": kept,
        "fieldEdits": [],
        "guide": None,
        # 5. 쟁점별 필수 안내 (예: 보증이면 민법 제428조의2 등). 표의 문구 그대로, LLM 없음.
        "notices": [f"{n.text} ({n.legal_ref})" for n in notices_for(issue_types)],
    }

# ======================================
# 작성 후 검증 (ClauseValidator가 사용)
# ======================================

NUMERIC_KINDS = ("MONTHS", "DAYS", "COUNT", "PERCENT", "AMOUNT", "DATE")


def tagged_values(text: str) -> set[tuple[str, Any]]:
    """
    글 안의 숫자 값을 종류를 붙여서 모두 꺼낸다.
    예: "2개월 연속, 연 7%" → {("PERIOD", (2, "개월")), ("PERCENT", 7.0)}
    """

    return (
        {("PERCENT", value) for value in find_percents(text)}
        | {("AMOUNT", value) for value in find_amounts(text)}
        | {("DATE", value) for value in find_dates(text)}
        | {("PERIOD", value) for value in _parse_periods(text)}
    )


def usable_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """규칙 검증에 쓸 수 있는 값 칸: 사용자가 확정했거나, 원문에 있는 값."""

    return [slot for slot in slots if slot.get("confirmed") or slot.get("inMessage")]


def check_slots_against_clause(
    *,
    slots: list[dict[str, Any]],
    message: str,
    clause: str,
) -> dict[str, Any]:
    usable = usable_slots(slots)
    clause_values = tagged_values(clause)

    not_reflected: list[str] = []
    explained: set[tuple[str, Any]] = set()
    old_values: set[tuple[str, Any]] = set()
    skip_fields: set[str] = set()

    for slot in usable:
        value_values = tagged_values(slot["value"])
        condition_values = tagged_values(slot.get("condition") or "")

        if slot["kind"] in NUMERIC_KINDS and (
            not value_values or not value_values <= clause_values
        ):
            not_reflected.append(slot["name"])

        if not condition_values <= clause_values:
            not_reflected.append(f"{slot['name']}(조건)")

        explained |= value_values | condition_values

        if slot.get("from"):
            from_values = tagged_values(slot["from"])
            old_values |= from_values
            explained |= from_values

        if slot.get("field") and slot.get("condition"):
            skip_fields.add(slot["field"])

    return {
        "usableCount": len(usable),
        "notReflected": not_reflected,
        "oldValues": old_values,
        "skipFields": skip_fields,
        "explainsMessage": bool(usable) and tagged_values(message) <= explained,
    }

