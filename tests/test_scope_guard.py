"""
범위 밖 요청(예: 김치찌개 만드는 법) 처리 테스트.

실제 OpenAI / Qdrant / 모델을 부르지 않도록
LLM 응답과 무거운 모듈을 가짜로 바꿔서 검증한다.
"""

import json
import os
import sys
import types

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

# law_retriever는 qdrant_client와 임베딩 모델(torch)을 불러오므로 가짜 모듈로 대체
fake_retriever_module = types.ModuleType("sai_rag.retrieval.law_retriever")


class FakeLawRetriever:  # noqa: D401 - 타입 자리만 채움
    pass


fake_retriever_module.LawRetriever = FakeLawRetriever
sys.modules["sai_rag.retrieval.law_retriever"] = fake_retriever_module

from sai_rag.agent.contract_clause_agent import ContractClauseAgent  # noqa: E402
from sai_rag.agent.issue_analyzer import ClauseIssueAnalyzer  # noqa: E402


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "lateInterestRate": 10,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
    "repaymentType": "LUMP_SUM",
}


class FakeResponses:
    def __init__(self, output: dict):
        self.output = output
        self.last_prompt = None
        self.last_kwargs = None

    def create(self, model, input, **kwargs):  # noqa: A002 - OpenAI SDK 인자 이름
        self.last_prompt = input
        self.last_kwargs = kwargs
        return types.SimpleNamespace(output_text=json.dumps(self.output, ensure_ascii=False))


def make_analyzer(output: dict) -> tuple[ClauseIssueAnalyzer, FakeResponses]:
    analyzer = ClauseIssueAnalyzer()
    fake = FakeResponses(output)
    analyzer.client = types.SimpleNamespace(responses=fake)
    return analyzer, fake


# ---------------------------------------------------------------
# 1. IssueAnalyzer
# ---------------------------------------------------------------

def test_analyzer_returns_empty_when_llm_says_out_of_scope():
    analyzer, _ = make_analyzer({"inScope": False, "issues": []})
    assert analyzer.analyze(CONTRACT, "김치찌개 만드는 법 알려줘") == []


def test_analyzer_ignores_issues_when_out_of_scope():
    # LLM이 실수로 쟁점을 같이 내도 inScope=false면 버린다
    analyzer, _ = make_analyzer({"inScope": False, "issues": [{"type": "OTHER", "reason": "x", "searchTerms": ["x"]}]})
    assert analyzer.analyze(CONTRACT, "오늘 날씨 어때?") == []


def test_analyzer_no_longer_forces_other_when_no_issue():
    # 예전 동작: 빈 결과면 OTHER 강제 추가 → 이제는 빈 목록
    analyzer, _ = make_analyzer({"issues": []})
    assert analyzer.analyze(CONTRACT, "아무 말") == []


def test_analyzer_keeps_in_scope_issues():
    analyzer, _ = make_analyzer(
        {
            "inScope": True,
            "issues": [
                {"type": "LATE_INTEREST", "reason": "연체이자", "searchTerms": ["지연손해금"]},
            ],
        }
    )
    issues = analyzer.analyze(CONTRACT, "연체 시 지연이자 특약 넣어줘")
    assert [issue["type"] for issue in issues] == ["LATE_INTEREST"]
    assert issues[0]["searchTerms"] == ["지연손해금"]


def test_prompt_contains_scope_rule():
    analyzer, fake = make_analyzer({"inScope": True, "issues": []})
    analyzer.analyze(CONTRACT, "김치찌개 만드는 법")
    assert '"inScope": false' in fake.last_prompt
    assert "요리" in fake.last_prompt


def test_analyzer_passes_call_options_to_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_ANALYZER_REASONING_EFFORT", "low")
    analyzer, fake = make_analyzer({"inScope": True, "issues": []})
    analyzer.analyze(CONTRACT, "연체 시 지연이자 특약 넣어줘")
    assert fake.last_kwargs["reasoning"] == {"effort": "low"}


def test_analyzer_sends_no_options_when_env_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_ANALYZER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENAI_ANALYZER_MAX_OUTPUT_TOKENS", raising=False)
    analyzer, fake = make_analyzer({"inScope": True, "issues": []})
    analyzer.analyze(CONTRACT, "연체 시 지연이자 특약 넣어줘")
    assert fake.last_kwargs == {}


# ---------------------------------------------------------------
# 2. ContractClauseAgent: 범위 밖이면 검색/리랭크/생성을 하지 않는다
# ---------------------------------------------------------------

class ExplodingDependency:
    """호출되면 테스트 실패 (범위 밖인데 무거운 단계까지 갔다는 뜻)"""

    def __getattr__(self, name):
        raise AssertionError(f"범위 밖 요청인데 '{name}'이(가) 호출됐습니다.")


def make_agent(analyzer_output: dict) -> ContractClauseAgent:
    agent = ContractClauseAgent(
        retriever=ExplodingDependency(),
        generator=ExplodingDependency(),
        reranker=ExplodingDependency(),
    )
    analyzer, _ = make_analyzer(analyzer_output)
    agent.issue_analyzer = analyzer
    return agent


@pytest.mark.parametrize(
    "message",
    ["김치찌개 만드는 법 알려줘", "오늘 서울 날씨 어때?", "부동산 매매계약서 써줘"],
)
def test_agent_returns_out_of_scope_without_heavy_steps(message):
    agent = make_agent({"inScope": False, "issues": []})

    result = agent.suggest_clause(contract=CONTRACT, message=message)

    assert result["validation"]["status"] == "OUT_OF_SCOPE"
    assert result["validation"]["isSafeToSuggest"] is False
    assert result["suggestedClause"] is None
    assert result["references"] == []
    assert result["debug"]["outOfScope"] is True


def test_out_of_scope_response_is_accepted_by_backend_rule():
    """
    백엔드 RagApiClient 규칙을 그대로 옮긴 검사:
    조항이 없으면 (isSafeToSuggest=false + 사유[message/issues/warnings])가 있어야 정상 응답.
    """
    agent = make_agent({"inScope": False, "issues": []})
    result = agent.suggest_clause(contract=CONTRACT, message="김치찌개 만드는 법")

    has_clause = bool(result["suggestedClause"])
    is_rejected = result["validation"]["isSafeToSuggest"] is False
    has_reason = bool(result["message"]) or bool(result["issues"]) or bool(result["warnings"])

    assert has_clause or (is_rejected and has_reason)
    # 프론트가 보여줄 안내 문구
    assert "금전소비대차" in result["summary"]
    assert result["warnings"]


def test_agent_in_scope_proceeds_to_retrieval():
    # 범위 안이면 기존처럼 다음 단계(검색)로 넘어간다 → 가짜 의존성이 호출되며 AssertionError
    agent = make_agent(
        {
            "inScope": True,
            "issues": [{"type": "LATE_INTEREST", "reason": "연체", "searchTerms": ["지연손해금"]}],
        }
    )
    with pytest.raises(AssertionError, match="호출됐습니다"):
        agent.suggest_clause(contract=CONTRACT, message="연체 시 지연이자 특약 넣어줘")