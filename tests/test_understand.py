"""
이해 확인(대화형 특약 1턴) 테스트.

쟁점 분석이 이해한 내용(understanding)과 정해야 할 것(questions)을
정리해서 돌려주는지, 검색·생성은 하지 않는지 확인한다.
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
from sai_rag.agent.issue_analyzer import ClauseIssueAnalyzer  # noqa: E402


CONTRACT = {
    "principal": 1_000_000,
    "interestRate": 5,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-01-31",
}

ISSUE = {"type": "ACCELERATION", "reason": "연체 시 일시 변제", "searchTerms": ["기한의 이익"]}


class FakeResponses:
    def __init__(self, output):
        self.output = output

    def create(self, **kwargs):
        return types.SimpleNamespace(output_text=json.dumps(self.output, ensure_ascii=False))


def make_analyzer(output):
    analyzer = ClauseIssueAnalyzer()
    analyzer.client = types.SimpleNamespace(responses=FakeResponses(output))
    return analyzer


class ExplodingDependency:
    """호출되면 테스트 실패 (이해 확인 단계인데 검색·생성까지 갔다는 뜻)"""

    def __getattr__(self, name):
        raise AssertionError(f"이해 확인 단계에서 '{name}'이(가) 호출됐습니다.")


def make_agent(output):
    agent = ContractClauseAgent(
        retriever=ExplodingDependency(),
        generator=ExplodingDependency(),
        reranker=ExplodingDependency(),
    )
    agent.issue_analyzer = make_analyzer(output)
    return agent


# ---------------------------------------------------------------
# 1. analyze_detailed: 이해한 내용과 질문 정리
# ---------------------------------------------------------------

def test_detailed_returns_understanding_and_questions():
    analyzer = make_analyzer(
        {
            "inScope": True,
            "issues": [ISSUE],
            "understanding": {
                "summary": "3개월 연속 연체 시 남은 원금을 한 번에 갚게 하는 조항을 원하시는 걸로 이해했어요.",
                "slots": [{"name": "연체 개월 수", "kind": "months", "value": "3개월"}],
            },
            "questions": [
                {"id": "partial_payment", "text": "일부만 낸 달도 연체로 볼까요?", "options": ["예", "아니요"]},
            ],
        }
    )

    result = analyzer.analyze_detailed(CONTRACT, "3개월 연속 연체하면 남은 원금을 전부 갚게 해 주세요")

    assert [issue["type"] for issue in result["issues"]] == ["ACCELERATION"]
    assert result["understanding"]["slots"] == [{"name": "연체 개월 수", "kind": "MONTHS", "value": "3개월"}]
    assert result["questions"][0]["options"] == ["예", "아니요"]


def test_unknown_slot_kind_becomes_other_and_bad_slots_are_dropped():
    analyzer = make_analyzer(
        {
            "inScope": True,
            "issues": [ISSUE],
            "understanding": {
                "summary": "요약",
                "slots": [
                    {"name": "연체 기간", "kind": "WEEKS", "value": "3주"},
                    {"name": "빈 값", "kind": "MONTHS", "value": ""},
                    "모양이 틀린 값",
                ],
            },
        }
    )

    slots = analyzer.analyze_detailed(CONTRACT, "요청")["understanding"]["slots"]

    assert slots == [{"name": "연체 기간", "kind": "OTHER", "value": "3주"}]


def test_questions_are_capped_and_deduplicated():
    raw_questions = [{"id": f"q{i}", "text": f"질문 {i}", "options": ["예"]} for i in range(5)]
    raw_questions.insert(1, {"id": "q0", "text": "같은 id", "options": []})
    raw_questions.insert(2, {"id": "", "text": "id 없음"})
    analyzer = make_analyzer({"inScope": True, "issues": [ISSUE], "questions": raw_questions})

    questions = analyzer.analyze_detailed(CONTRACT, "요청")["questions"]

    assert [question["id"] for question in questions] == ["q0", "q1", "q2"]
    assert questions[0]["text"] == "질문 0"


def test_options_that_are_not_a_list_become_empty():
    analyzer = make_analyzer(
        {
            "inScope": True,
            "issues": [ISSUE],
            "questions": [{"id": "guarantor", "text": "보증인을 둘까요?", "options": "예"}],
        }
    )

    questions = analyzer.analyze_detailed(CONTRACT, "요청")["questions"]

    assert questions[0]["options"] == []


def test_out_of_scope_has_no_understanding_or_questions():
    analyzer = make_analyzer(
        {
            "inScope": False,
            "issues": [],
            "understanding": {"summary": "x"},
            "questions": [{"id": "x", "text": "x"}],
        }
    )

    result = analyzer.analyze_detailed(CONTRACT, "김치찌개 만드는 법")

    assert result == {"issues": [], "understanding": None, "questions": []}


def test_analyze_still_returns_only_issues():
    analyzer = make_analyzer({"inScope": True, "issues": [ISSUE], "questions": [{"id": "x", "text": "x"}]})

    assert analyzer.analyze(CONTRACT, "요청") == [ISSUE]


# ---------------------------------------------------------------
# 2. ContractClauseAgent.understand: 검색·생성 없이 이해 확인만
# ---------------------------------------------------------------

def test_understand_does_not_search_or_write():
    agent = make_agent(
        {
            "inScope": True,
            "issues": [ISSUE],
            "understanding": {"summary": "요약", "slots": []},
            "questions": [],
        }
    )

    result = agent.understand(contract=CONTRACT, message="3개월 연속 연체하면 전부 갚게 해 주세요")

    assert result["inScope"] is True
    assert result["understanding"]["summary"] == "요약"
    assert result["issues"][0]["searchTerms"] == ["기한의 이익"]
    assert "issueAnalysisMs" in result["debug"]["performance"]


def test_understand_out_of_scope_returns_guide():
    agent = make_agent({"inScope": False, "issues": []})

    result = agent.understand(contract=CONTRACT, message="김치찌개 만드는 법")

    assert result["inScope"] is False
    assert result["understanding"] is None
    assert "금전소비대차" in result["guide"]