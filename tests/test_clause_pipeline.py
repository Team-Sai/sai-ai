"""
특약 생성 전체 흐름(①~⑤) 안전망 테스트.

suggest_clause()를 단계별 함수로 나눌 때
이벤트 순서와 응답 모양이 바뀌지 않았는지 확인한다.
OpenAI, Qdrant, 모델은 모두 가짜로 바꾼다.
"""

import json
import os
import sys
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

# law_retriever는 qdrant_client와 임베딩 모델을 불러오므로 가짜 모듈로 대체
fake_retriever_module = types.ModuleType("sai_rag.retrieval.law_retriever")
fake_retriever_module.LawRetriever = object
sys.modules.setdefault("sai_rag.retrieval.law_retriever", fake_retriever_module)

from sai_rag.agent.contract_clause_agent import ContractClauseAgent  # noqa: E402


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "lateInterestRate": 10,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
    "repaymentType": "LUMP_SUM",
}

ANALYZER_OUTPUT = {
    "inScope": True,
    "issues": [
        {"type": "EARLY_REPAYMENT", "reason": "중도상환", "searchTerms": ["중도상환 수수료"]},
    ],
}

LAW = {
    "document_id": "민법-제468조",
    "score": 0.9,
    "text": "테스트용 조문 본문",
    "metadata": {"law_name": "민법", "article_number": "제468조", "article_title": "변제기전의 변제"},
}

CLAUSE = "제1조 채무자는 언제든지 중도상환할 수 있다."


class FakeAnalyzerResponses:
    def create(self, **kwargs):
        return types.SimpleNamespace(output_text=json.dumps(ANALYZER_OUTPUT, ensure_ascii=False))


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, question, top_k, law_names):
        self.calls += 1
        return [dict(LAW)]


class FakeReranker:
    def rerank(self, candidates, top_k):
        return [{**candidate, "rerank_score": 0.8} for candidate in candidates]


class FakeHybridRanker:
    rerank_weight = 0.7
    vector_weight = 0.3

    def rank(self, candidates, top_k):
        return [{**candidate, "final_score": 1.0} for candidate in candidates][:top_k]


class FakeGenerator:
    def generate(self, contract, message, issues, references, on_delta=None, should_stop=None, **kwargs):
        self.last_kwargs = kwargs
        if on_delta is not None:
            on_delta("제1조 ")
        return {
            "title": "중도상환",
            "summary": "요약",
            "suggestedClause": CLAUSE,
            "warnings": [],
            "unresolvedDecisions": [],
        }


class FakeValidator:
    def validate(self, contract, message, issues, generated, references, should_stop=None, **kwargs):
        self.last_kwargs = kwargs
        return {"status": "PASS", "isSafeToSuggest": True, "checks": [], "validatorMode": "DETERMINISTIC"}


def make_agent():
    retriever = FakeRetriever()
    agent = ContractClauseAgent(retriever=retriever, generator=FakeGenerator(), reranker=FakeReranker())
    agent.issue_analyzer.client = types.SimpleNamespace(responses=FakeAnalyzerResponses())
    agent.hybrid_ranker = FakeHybridRanker()
    agent.validator = FakeValidator()
    return agent, retriever


def run(agent):
    events = []
    result = agent.suggest_clause(
        contract=CONTRACT,
        message="중도상환 수수료 없이 해 주세요",
        on_event=lambda event, data: events.append((event, data)),
    )
    return result, events


def test_full_pipeline_event_order():
    agent, _ = make_agent()
    _, events = run(agent)

    names = [event for event, _ in events]
    steps = [data["step"] for event, data in events if event == "progress"]

    assert names == ["progress", "progress", "references", "references", "progress", "delta", "draft", "progress"]
    assert steps == ["analysis", "retrieval", "generation", "validation"]


def test_full_pipeline_response_shape():
    agent, retriever = make_agent()
    result, _ = run(agent)

    assert set(result) == {
        "message", "issues", "title", "summary", "suggestedClause",
        "warnings", "validation", "references", "unresolvedDecisions", "debug",
    }
    assert set(result["debug"]["performance"]) == {
        "issueAnalysisMs", "queryPlanningMs", "retrievalMs", "candidateMergeMs", "rerankingMs",
        "hybridRankingMs", "generationMs", "validationMs", "totalMs", "retrievalDetails",
    }
    assert result["suggestedClause"] == CLAUSE
    assert result["validation"]["status"] == "PASS"
    assert result["debug"]["validatorMode"] == "DETERMINISTIC"
    assert result["references"][0]["lawName"] == "민법"
    assert retriever.calls == len(result["debug"]["searchPlans"])

WRITE_SLOT = {
    "name": "연속 연체 기간",
    "kind": "MONTHS",
    "value": "3개월",
    "from": None,
    "field": None,
    "condition": None,
    "inMessage": True,
    "confirmed": True,
}


class ExplodingAnalyzerResponses:
    def create(self, **kwargs):
        raise AssertionError("작성 단계에서 쟁점 분석(LLM)을 다시 불렀습니다.")


def test_write_with_given_issues_skips_analysis_and_passes_confirmed_values():
    agent, _ = make_agent()
    agent.issue_analyzer.client = types.SimpleNamespace(responses=ExplodingAnalyzerResponses())
    events = []

    result = agent.suggest_clause(
        contract=CONTRACT,
        message="중도상환 수수료 없이 해 주세요",
        on_event=lambda event, data: events.append((event, data)),
        issues=ANALYZER_OUTPUT["issues"],
        slots=[WRITE_SLOT],
        questions=[],
        answers={},
    )

    steps = [data["step"] for event, data in events if event == "progress"]

    assert steps == ["retrieval", "generation", "validation"]
    assert agent.generator.last_kwargs["confirmed"]["slots"] == [WRITE_SLOT]
    assert agent.validator.last_kwargs["slots"] == [WRITE_SLOT]
    assert result["debug"]["performance"]["issueAnalysisMs"] == 0.0


def test_guarantee_notices_are_always_added_to_chat_warnings():
    agent, _ = make_agent()

    result = agent.suggest_clause(
        contract=CONTRACT,
        message="형이 보증인이 되게 해 주세요",
        issues=[{"type": "GUARANTEE", "reason": "보증", "searchTerms": ["보증"]}],
        slots=[],
        questions=[],
        answers={},
    )

    assert any("민법 제428조의2제1항" in warning for warning in result["warnings"])
    assert any("보증인 보호를 위한 특별법 제4조" in warning for warning in result["warnings"])

