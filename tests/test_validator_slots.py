"""
값 칸(slots)을 받은 검증기 테스트.

확정됐거나 원문에 있는 값 칸이 조항에 반영됐으면
조건 있는 계약 값 변경도 LLM 없이 규칙으로 끝나는지,
아니면 지금처럼 LLM으로 가는지 확인한다.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from sai_rag.agent.clause_validator import ClauseValidator  # noqa: E402


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "lateInterestRate": 10,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
    "repaymentType": "LUMP_SUM",
}

ISSUES = [{"type": "INTEREST"}]
MESSAGE = "2개월 연속 연체하면 이자율을 연 5%에서 연 7%로 올려 주세요"
CLAUSE = "제1조(이자율 변경) 채무자가 2개월 연속하여 연체한 경우 그 다음 달부터 약정 이자율은 연 7%로 한다."

REFERENCES = [
    {"lawName": "민법", "articleNo": "제397조", "text": "테스트용 조문 본문", "score": 0.9},
]

SLOT = {
    "name": "연체 시 이자율",
    "kind": "PERCENT",
    "value": "연 7%",
    "from": "연 5%",
    "field": "interestRate",
    "condition": "2개월 연속 연체 시",
    "inMessage": True,
    "confirmed": True,
}


class LlmCalled(Exception):
    pass


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise LlmCalled("LLM 검증이 호출되었습니다")


@pytest.fixture
def validator():
    v = ClauseValidator(api_key="test-key-not-real")
    v.client = type("FakeClient", (), {"responses": RecordingResponses()})()
    return v


def run(validator, clause=CLAUSE, slots=None):
    try:
        return validator.validate(
            message=MESSAGE,
            contract=CONTRACT,
            issues=ISSUES,
            generated={"suggestedClause": clause, "warnings": [], "unresolvedDecisions": []},
            references=REFERENCES,
            slots=slots,
        )
    except LlmCalled:
        return None


def test_without_slots_conditioned_rate_change_goes_to_llm(validator):
    # 지금 동작: 계약 5%와 다른 7%, 원문의 5% 미반영, "연속" 표시어 때문에 LLM으로 간다
    run(validator)

    assert len(validator.client.responses.calls) == 1


def test_confirmed_slot_finishes_by_rule(validator):
    result = run(validator, slots=[SLOT])

    assert validator.client.responses.calls == []
    assert result["validatorMode"] == "DETERMINISTIC"
    assert result["isSafeToSuggest"] is True


def test_slot_value_missing_from_clause_goes_to_llm(validator):
    clause = CLAUSE.replace("연 7%", "연 6%")

    run(validator, clause=clause, slots=[SLOT])

    assert len(validator.client.responses.calls) == 1


def test_unconfirmed_slot_not_in_message_is_not_trusted(validator):
    slot = {**SLOT, "inMessage": False, "confirmed": False}

    run(validator, slots=[slot])

    assert len(validator.client.responses.calls) == 1


def test_marker_missing_in_clause_goes_to_llm(validator):
    # 원문은 "연속"인데 조항이 "이상"으로 바꿔 썼으면 뜻이 다를 수 있어 LLM으로 보낸다
    clause = CLAUSE.replace("연속하여", "이상")

    run(validator, clause=clause, slots=[SLOT])

    assert len(validator.client.responses.calls) == 1


def test_chat_conditions_go_into_llm_prompt_and_replace_candidates(validator):
    clause = "채무자가 연체하면 [입력 필요: 연체이자 계산 대상]에 대해 연 12%의 지연이자를 지급한다."

    try:
        validator.validate(
            message="연체하면 연 12%의 지연이자를 받고 싶어요",
            contract=CONTRACT,
            issues=[{"type": "LATE_INTEREST"}, {"type": "OTHER"}],
            generated={"suggestedClause": clause, "warnings": [], "unresolvedDecisions": []},
            references=REFERENCES,
            slots=[],
            confirmed={"slots": [], "questions": [], "answers": {}},
        )
    except LlmCalled:
        pass

    prompt = validator.client.responses.calls[0]["input"]

    assert "대화에서 이미 정한 조건" in prompt
    assert "연체이자는 연체된 금액에 대해 계산한다" in prompt
    assert "[해당 쟁점의 결정사항 후보]\n[]" in prompt


def test_vague_rate_phrase_is_warned_by_rule(validator):
    result = run(validator, clause=CLAUSE + " 위약금은 연체된 금액에 일정 비율을 적용하여 산정한다.", slots=[SLOT])

    assert validator.client.responses.calls == []
    ambiguity = next(check for check in result["checks"] if check["type"] == "AMBIGUITY")
    assert ambiguity["status"] == "WARNING"


def test_slot_used_as_basis_is_warned_by_rule(validator):
    clause = CLAUSE + " 위약금은 [입력 필요: 위약금 산정 기준]에 따라 산정한다."

    result = run(validator, clause=clause, slots=[SLOT])

    ambiguity = next(check for check in result["checks"] if check["type"] == "AMBIGUITY")
    assert ambiguity["status"] == "WARNING"

