"""
사용자 답장 규칙 분류 테스트.

지시서 4-3의 "걸리면 안 되는 말"을 포함해,
확실할 때만 분류하고 애매하면 AMBIGUOUS인지 확인한다.
"""

import pytest

from sai_rag.agent.chat_router import UNKNOWN_ANSWER, route_reply


CONTRACT = {"interestRate": 5}

PERIOD_SLOT = {
    "name": "연속 연체 기간",
    "kind": "MONTHS",
    "value": "3개월",
    "from": None,
    "field": None,
    "condition": None,
    "inMessage": True,
    "confirmed": False,
}

RATE_SLOT = {
    "name": "연체 시 이자율",
    "kind": "PERCENT",
    "value": "7%",
    "from": "6%",
    "field": "interestRate",
    "condition": "2개월 연속 연체 시",
    "inMessage": True,
    "confirmed": False,
}

PARTIAL_QUESTION = {"id": "partial_payment", "text": "일부만 낸 달도 연체로 볼까요?", "options": ["예", "아니요"]}
SCOPE_QUESTION = {"id": "scope", "text": "어디에 적용할까요?", "options": ["연체 금액", "남은 원금"]}
FROM_QUESTION = {
    "id": "confirm_from_interestRate",
    "text": "지금 계약 이자율은 5%예요. 5%에서 7%로 바꾸는 게 맞을까요?",
    "options": ["맞아요", "아니요"],
}


def run(message, slots=None, questions=(), answers=None):
    return route_reply(
        message=message,
        contract=CONTRACT,
        slots=slots if slots is not None else [PERIOD_SLOT],
        questions=list(questions),
        answers=answers or {},
    )


# ---------------------------------------------------------------
# 1. 확정
# ---------------------------------------------------------------

def test_single_confirm_word_confirms_all_slots():
    result = run("네!")

    assert result["kind"] == "CONFIRM"
    assert result["readyToWrite"] is True
    assert result["slots"][0]["confirmed"] is True


def test_answered_question_is_not_asked_again():
    result = run("네", questions=[PARTIAL_QUESTION], answers={"partial_payment": "예"})

    assert result["kind"] == "CONFIRM"
    assert result["pendingQuestions"] == []


# ---------------------------------------------------------------
# 2. 값 바꾸기 (4-3 걸리면 안 되는 말 포함)
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("예전처럼 3개월로 해 줘", "3개월"),
        ("네 달로 해 줘", "4개월"),
        ("아니, 2개월로", "2개월"),
    ],
)
def test_value_change(message, expected):
    result = run(message)

    assert result["kind"] == "VALUE_CHANGE"
    assert result["slots"][0]["value"] == expected
    assert result["slots"][0]["confirmed"] is True
    assert expected in result["echo"]


def test_from_to_uses_target_value():
    result = run("5%에서 7%로", slots=[{**RATE_SLOT, "value": "6%"}])

    assert result["kind"] == "VALUE_CHANGE"
    assert result["slots"][0]["value"] == "7%"


def test_input_slots_are_not_changed():
    slots = [dict(PERIOD_SLOT)]

    run("2개월로", slots=slots)

    assert slots[0]["value"] == "3개월"
    assert slots[0]["confirmed"] is False


@pytest.mark.parametrize(
    "message",
    [
        "2개월 이상이면",
        "응 근데 이자도 좀 낮춰 줘",
        "'즉시'를 '7일 이내'로 바꿔 줘",
    ],
)
def test_ambiguous_messages_change_nothing(message):
    result = run(message)

    assert result["kind"] == "AMBIGUOUS"
    assert result["slots"][0]["value"] == "3개월"
    assert result["slots"][0]["confirmed"] is False


# ---------------------------------------------------------------
# 3. 질문에 답하기
# ---------------------------------------------------------------

def test_button_answer_is_recorded():
    result = run("아니요", questions=[PARTIAL_QUESTION])

    assert result["kind"] == "ANSWER"
    assert result["answers"] == {"partial_payment": "아니요"}
    assert result["pendingQuestions"] == []


def test_confirm_word_answers_single_question():
    result = run("네", questions=[PARTIAL_QUESTION])

    assert result["kind"] == "ANSWER"
    assert result["answers"] == {"partial_payment": "예"}


def test_confirm_word_with_two_questions_is_ambiguous():
    result = run("네", questions=[PARTIAL_QUESTION, SCOPE_QUESTION])

    assert result["kind"] == "AMBIGUOUS"
    assert result["answers"] == {}

def test_button_answer_works_with_two_questions():
    result = run("남은 원금", questions=[PARTIAL_QUESTION, SCOPE_QUESTION])

    assert result["kind"] == "ANSWER"
    assert result["answers"] == {"scope": "남은 원금"}
    assert [question["id"] for question in result["pendingQuestions"]] == ["partial_payment"]


def test_unknown_answer_is_recorded():
    result = run("나중에 정할게요", questions=[PARTIAL_QUESTION])

    assert result["kind"] == "UNKNOWN"
    assert result["answers"] == {"partial_payment": UNKNOWN_ANSWER}


def test_from_question_yes_uses_contract_value():
    result = run("맞아요", slots=[RATE_SLOT], questions=[FROM_QUESTION])

    assert result["kind"] == "ANSWER"
    assert result["slots"][0]["from"] == "5%"


def test_from_question_no_is_reject():
    result = run("아니요", slots=[RATE_SLOT], questions=[FROM_QUESTION])

    assert result["kind"] == "REJECT"
    assert result["slots"][0]["from"] == "6%"