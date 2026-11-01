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


def _ro(word: str) -> str:
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
            f"{current}에서 {value}{_ro(value)} 바꾸는 게 맞을까요?"
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


def apply_slot_rules(
    *,
    contract: dict[str, Any],
    message: str,
    understanding: dict[str, Any],
    questions: list[dict[str, Any]],
    max_questions: int,
) -> dict[str, Any]:
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
        }

    rule_questions: list[dict[str, Any]] = []

    for slot in slots:
        question = build_from_question(slot, contract)

        if question and question["id"] not in {q["id"] for q in rule_questions}:
            rule_questions.append(question)

    rule_ids = {question["id"] for question in rule_questions}
    merged = rule_questions + [question for question in questions if question["id"] not in rule_ids]

    return {
        "understanding": understanding,
        "questions": merged[:max_questions],
        "fieldEdits": [],
        "guide": None,
    }