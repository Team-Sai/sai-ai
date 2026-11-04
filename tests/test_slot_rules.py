"""
이해 단계 값 칸 규칙 테스트.

원문 확인, 기존 값 확인 질문, 본문 칸 안내가
확실한 경우에만 동작하고, 모르면 아무것도 하지 않는지 확인한다.
"""

from sai_rag.agent.slot_rules import apply_slot_rules, check_slots_against_clause

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


def run(message, slots, questions=None, issue_types=()):
    return apply_slot_rules(
        contract=CONTRACT,
        message=message,
        understanding={"summary": "요약", "slots": slots},
        questions=questions or [],
        max_questions=3,
        issue_types=issue_types,
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

# ---------------------------------------------------------------
# 4. 작성 후 검증용 비교
# ---------------------------------------------------------------

RATE_MESSAGE = "2개월 연속 연체하면 이자율을 연 5%에서 연 7%로 올려 주세요"
RATE_CLAUSE = "채무자가 2개월 연속하여 연체한 경우 그 다음 달부터 약정 이자율은 연 7%로 한다."


def rate_slot(**overrides):
    slot = make_slot(value="연 7%", field="interestRate", condition="2개월 연속 연체 시", confirmed=True)
    slot["from"] = "연 5%"
    slot.update(overrides)
    return slot


def test_reflected_conditioned_change_explains_message():
    result = check_slots_against_clause(slots=[rate_slot()], message=RATE_MESSAGE, clause=RATE_CLAUSE)

    assert result["notReflected"] == []
    assert result["skipFields"] == {"interestRate"}
    assert ("PERCENT", 5.0) in result["oldValues"]
    assert result["explainsMessage"] is True


def test_missing_value_or_condition_is_reported():
    clause = "채무자가 3개월 연속하여 연체한 경우 약정 이자율은 연 6%로 한다."

    result = check_slots_against_clause(slots=[rate_slot()], message=RATE_MESSAGE, clause=clause)

    assert result["notReflected"] == ["값", "값(조건)"]


def test_unconfirmed_slot_not_in_message_is_not_used():
    slot = rate_slot(confirmed=False, inMessage=False)

    result = check_slots_against_clause(slots=[slot], message=RATE_MESSAGE, clause=RATE_CLAUSE)

    assert result["usableCount"] == 0
    assert result["skipFields"] == set()
    assert result["explainsMessage"] is False


def test_unexplained_number_in_message_blocks_rule_path():
    message = RATE_MESSAGE + " 그리고 위약금은 10만원으로 해 주세요"

    result = check_slots_against_clause(slots=[rate_slot()], message=message, clause=RATE_CLAUSE)

    assert result["explainsMessage"] is False




# ---------------------------------------------------------------
# 5. 쟁점별 필수 질문 · 기본값 · 필수 안내
# ---------------------------------------------------------------

def question_ids(result):
    return [question["id"] for question in result["questions"]]


def test_required_question_is_added_when_not_covered():
    result = run("3개월 연속 연체하면 전부 갚게 해 주세요", [], issue_types=["DEFAULT"])

    assert question_ids(result) == ["partial_payment", "confirm_defaults"]
    assert "연체 횟수는 회차별로 센다" in result["questions"][-1]["text"]
    assert result["questions"][-1]["options"] == ["좋아요", "바꿀래요"]


def test_required_question_is_skipped_when_user_already_said_it():
    result = run("일부만 내도 연체로 보고 3개월 연속이면 전부 갚게", [], issue_types=["DEFAULT"])

    assert question_ids(result) == ["confirm_defaults"]


def test_required_question_is_skipped_when_llm_asked_it():
    llm = [{"id": "llm_partial", "text": "일부만 낸 달도 연체로 볼까요?", "options": ["예", "아니요"]}]

    result = run("3개월 연속 연체하면 전부 갚게", [], llm, issue_types=["DEFAULT"])

    assert question_ids(result) == ["llm_partial", "confirm_defaults"]


def test_question_without_default_wins_and_overflow_becomes_default():
    llm = [{"id": f"q{i}", "text": f"질문 {i}", "options": []} for i in range(2)]

    result = run("약속을 어기면 위약금 50만원", [], llm, issue_types=["PENALTY"])

    # 기본값 없는 질문(위약금과 연체이자) → LLM 질문 2개 → (상한 3개) → 기본값 있는 질문(발생 횟수)은 밀려남
    assert question_ids(result) == ["penalty_with_late_interest", "q0", "q1", "confirm_defaults"]
    assert "위약금은 한 번만 받는다" in result["questions"][-1]["text"]


def test_guarantee_has_limit_question_and_legal_notices():
    result = run("채무자의 형이 보증인이 되는 내용을 넣고 싶어요", [], issue_types=["GUARANTEE"])

    assert question_ids(result)[0] == "guarantee_limit"
    assert any("민법 제428조의2제1항" in notice for notice in result["notices"])
    assert any("보증인 보호를 위한 특별법 제4조" in notice for notice in result["notices"])

