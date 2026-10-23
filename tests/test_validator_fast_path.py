"""
ClauseValidator 속도 개선 테스트.

- 모호 표현만 있는 조항은 LLM 검증 없이 Fast Path로 끝나고 WARNING으로 안내한다.
- 깨끗한 조항은 기존처럼 PASS.
- 계약 확정값과 충돌하는 조항은 여전히 LLM 검증으로 넘어간다.
- 호출 옵션 환경변수는 설정했을 때만 붙는다.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from sai_rag.agent.clause_validator import ClauseValidator  # noqa: E402
from sai_rag.agent.openai_options import openai_call_options  # noqa: E402


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "lateInterestRate": 10,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
    "repaymentType": "LUMP_SUM",
}

ISSUES = [{"type": "EARLY_REPAYMENT"}]
MESSAGE = "중도상환 수수료 없이 미리 갚을 수 있게 해줘"

CLEAN_CLAUSE = (
    "제1조(중도상환) 채무자는 상환기일 전이라도 대여금을 "
    "중도상환수수료 없이 상환할 수 있다."
)
AMBIGUOUS_CLAUSE = (
    "제1조(중도상환) 채무자는 상환기일 전이라도 상당한 기간 전에 채권자에게 "
    "통지한 후 대여금을 중도상환수수료 없이 상환할 수 있다."
)

REFERENCES = [
    {
        "lawName": "민법",
        "articleNo": "제468조",
        "text": "당사자의 특별한 의사표시가 없으면 변제기 전이라도 채무자는 변제할 수 있다.",
        "score": 0.9,
    }
]

CONFLICT_CLAUSE = (
    "제1조(중도상환) 채무자는 대여금 2,000,000원을 상환기일 전이라도 "
    "중도상환수수료 없이 상환할 수 있다."
)


class LlmCalled(Exception):
    pass


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise LlmCalled("LLM 검증이 호출되었습니다")


class FakeClient:
    def __init__(self):
        self.responses = RecordingResponses()


@pytest.fixture
def validator():
    v = ClauseValidator(api_key="test-key-not-real")
    v.client = FakeClient()
    return v


def run(validator, clause, message=MESSAGE):
    return validator.validate(
        message=message,
        contract=CONTRACT,
        issues=ISSUES,
        generated={"suggestedClause": clause, "warnings": [], "unresolvedDecisions": []},
        references=REFERENCES,
    )


def test_clean_clause_passes_without_llm(validator):
    result = run(validator, CLEAN_CLAUSE)
    assert validator.client.responses.calls == []
    assert result["status"] == "PASS"
    assert result["isSafeToSuggest"] is True


def test_ambiguous_phrase_is_warning_not_llm(validator):
    result = run(validator, AMBIGUOUS_CLAUSE)

    # LLM을 부르지 않는다
    assert validator.client.responses.calls == []

    # 반영은 가능하지만 주의 필요
    assert result["status"] == "WARNING"
    assert result["isSafeToSuggest"] is True

    ambiguity = [c for c in result["checks"] if c["type"] == "AMBIGUITY"]
    assert ambiguity and ambiguity[0]["status"] == "WARNING"
    assert "상당한 기간" in ambiguity[0]["message"]
    assert any("상당한 기간" in w for w in result["warnings"])


def test_contract_conflict_still_goes_to_llm(validator):
    # 원금이 계약(100만 원)과 다른 조항은 안전장치로 LLM 검증까지 가야 한다
    try:
        result = run(validator, CONFLICT_CLAUSE)
    except LlmCalled:
        result = None

    assert len(validator.client.responses.calls) == 1
    if result is not None:
        assert result["status"] != "PASS"


def test_call_options_empty_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_VALIDATOR_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_VALIDATOR_MAX_OUTPUT_TOKENS", raising=False)
    assert openai_call_options("VALIDATOR") == {}


def test_call_options_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_GENERATOR_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_GENERATOR_MAX_OUTPUT_TOKENS", "2000")
    assert openai_call_options("GENERATOR") == {
        "reasoning": {"effort": "low"},
        "max_output_tokens": 2000,
    }


def test_validator_passes_call_options_to_llm(validator, monkeypatch):
    monkeypatch.setenv("OPENAI_VALIDATOR_REASONING_EFFORT", "low")
    try:
        run(validator, CONFLICT_CLAUSE)
    except LlmCalled:
        pass
    assert validator.client.responses.calls
    assert validator.client.responses.calls[0]["reasoning"] == {"effort": "low"}



@pytest.mark.parametrize("bad_value", ["abc", "2,000", "0", "-5", "1.5"])
def test_invalid_max_output_tokens_is_ignored(monkeypatch, bad_value):
    monkeypatch.delenv("OPENAI_GENERATOR_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("OPENAI_GENERATOR_MAX_OUTPUT_TOKENS", bad_value)
    # 예외 없이 옵션만 빠져야 한다
    assert openai_call_options("GENERATOR") == {}


def test_invalid_max_output_tokens_keeps_other_options(monkeypatch):
    monkeypatch.setenv("OPENAI_VALIDATOR_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENAI_VALIDATOR_MAX_OUTPUT_TOKENS", "abc")
    assert openai_call_options("VALIDATOR") == {"reasoning": {"effort": "low"}}



def test_cancel_stops_before_llm_validation(validator):
    # 사용자가 취소했으면, LLM 검증(약 12초)이 필요한 조항이어도 부르기 전에 멈춘다
    from sai_rag.agent.clause_generator import ClauseGenerationCancelled

    with pytest.raises(ClauseGenerationCancelled):
        validator.validate(
            message=MESSAGE,
            contract=CONTRACT,
            issues=ISSUES,
            generated={"suggestedClause": CONFLICT_CLAUSE, "warnings": [], "unresolvedDecisions": []},
            references=REFERENCES,
            should_stop=lambda: True,
        )

    assert validator.client.responses.calls == []
