"""
이해 단계 값 칸 규칙 테스트.

원문 확인, 기존 값 확인 질문, 본문 칸 안내가
확실한 경우에만 동작하고, 모르면 아무것도 하지 않는지 확인한다.
"""

from sai_rag.agent.slot_rules import apply_slot_rules


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "lateInterestRate": None,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
    "repaymentDay": 25,
    "repaymentType": "EQUAL_PRINCIPAL_AND_INTEREST",
}


def make_slot(**overrides):
    slot = {
        "name": "값",
        "kind": "PERCENT",
        "value": "7%",
        "from": None,
        "field": None,
        "condition": None,
        "inMessage": False,
        "confirmed": False,
    }
    slot.update(overrides)
    return slot


def run(message, slots, questions=None):
    return apply_slot_rules(
        contract=CONTRACT,
        message=message,
        understanding={"summary": "요약", "slots": slots},
        questions=questions or [],
        max_questions=3,
    )


# ---------------------------------------------------------------
# 1. 원문 확인
# ---------------------------------------------------------------

def test_value_in_message_is_marked():
    slot = make_slot(value="7%", from_="5%")
    slot = make_slot(value="7%")

    run("2개월 연속 연체하면 이자율을 5%에서 7%로 올려 주세요", [slot])

    assert slot["inMessage"] is True


def test_inferred_value_is_not_marked():
    slot = make_slot(value="6%")

    run("이자율을 1%p 올려 주세요", [slot])

    assert slot["inMessage"] is False


def test_period_values_are_read_but_korean_numbers_are_not():
    in_digits = make_slot(kind="MONTHS", value="3개월 연속")
    in_korean = make_slot(kind="MONTHS", value="3개월")

    run("3개월 연속 연체하면 전부 갚게 해 주세요", [in_digits])
    run("세 달 연속 연체하면 전부 갚게 해 주세요", [in_korean])

    assert in_digits["inMessage"] is True
    assert in_korean["inMessage"] is False


# ---------------------------------------------------------------
# 2. 기존 값 확인 질문
# ---------------------------------------------------------------

def test_wrong_from_value_becomes_question():
    slot = make_slot(value="7%", field="interestRate", condition="2개월 연속 연체 시")
    slot["from"] = "6%"

    result = run("2개월 연속 연체하면 이자율을 6%에서 7%로 올려 주세요", [slot])

    assert result["questions"][0] == {
        "id": "confirm_from_interestRate",
        "text": "지금 계약 이자율은 5%예요. 5%에서 7%로 바꾸는 게 맞을까요?",
        "options": ["맞아요", "아니요"],
    }


def test_question_uses_correct_particles_for_amounts():
    slot = make_slot(kind="AMOUNT", value="300만원", field="principalAmount", condition="보증인이 생기면")
    slot["from"] = "200만원"

    result = run("보증인이 생기면 원금을 200만원에서 300만원으로 늘려 주세요", [slot])

    assert result["questions"][0]["text"] == (
        "지금 계약 원금은 1,000,000원이에요. 1,000,000원에서 300만원으로 바꾸는 게 맞을까요?"
    )


def test_matching_from_value_asks_nothing():
    slot = make_slot(value="7%", field="interestRate", condition="2개월 연속 연체 시")
    slot["from"] = "5%"

    result = run("2개월 연속 연체하면 이자율을 5%에서 7%로 올려 주세요", [slot])

    assert result["questions"] == []


def test_rule_question_comes_first_and_questions_are_capped():
    slot = make_slot(value="7%", field="interestRate", condition="2개월 연속 연체 시")
    slot["from"] = "6%"
    llm_questions = [{"id": f"q{i}", "text": f"질문 {i}", "options": []} for i in range(3)]

    result = run("2개월 연속 연체하면 6%에서 7%로", [slot], llm_questions)

    assert [question["id"] for question in result["questions"]] == ["confirm_from_interestRate", "q0", "q1"]


# ---------------------------------------------------------------
# 3. 본문 칸 안내
# ---------------------------------------------------------------

def test_unconditioned_field_change_becomes_guide():
    slot = make_slot(value="7%", field="interestRate")

    result = run("이자율을 7%로 바꿔 주세요", [slot], [{"id": "x", "text": "x", "options": []}])

    assert result["fieldEdits"] == [
        {"field": "interestRate", "label": "이자율", "value": "7%", "current": "5%"},
    ]
    assert "이자율 칸을 바꾸면" in result["guide"]
    assert result["questions"] == []


def test_same_value_as_contract_is_not_a_change():
    slot = make_slot(value="5%", field="interestRate")

    result = run("이자율 5%는 그대로 두고 연체 조건만 넣어 주세요", [slot])

    assert result["fieldEdits"] == []
    assert result["guide"] is None


def test_late_interest_rate_is_always_a_special_clause():
    slot = make_slot(value="12%", field="lateInterestRate")

    result = run("연체이율을 12%로 해 주세요", [slot])

    assert result["fieldEdits"] == []
    assert result["guide"] is None