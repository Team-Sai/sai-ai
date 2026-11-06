import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from sai_rag.agent import demo_scenarios  # noqa: E402
from tests.test_clause_pipeline import (  # noqa: E402
    CONTRACT,
    ExplodingAnalyzerResponses,
    make_agent,
)

QUICK = "3개월 연속 연체하면 남은 원금을 전부 갚게 하고 싶어요."


def test_no_demo_without_server_switch(monkeypatch):
    monkeypatch.delenv(demo_scenarios.DEMO_ENV, raising=False)

    assert demo_scenarios.find_demo_scenario(QUICK, demo=True) is None


def test_no_demo_without_request_flag(monkeypatch):
    monkeypatch.setenv(demo_scenarios.DEMO_ENV, "1")

    assert demo_scenarios.find_demo_scenario(QUICK, demo=False) is None


def test_demo_finds_all_four_quick_questions(monkeypatch):
    monkeypatch.setenv(demo_scenarios.DEMO_ENV, "1")

    for question in (
        "3개월 연속 연체하면 남은 원금을 전부 갚게 하고 싶어요",
        "만기 전에 원금을 먼저 갚을 수 있게 하고 싶어요.",
        "연체했을 때 위약금을 정하고 싶어요.",
        "채무자 소유 자동차를 담보로 잡고 싶어요.",
    ):
        assert demo_scenarios.find_demo_scenario(question, demo=True) is not None

    assert demo_scenarios.find_demo_scenario("다른 질문이에요", demo=True) is None


def test_demo_understand_skips_llm_but_uses_real_rules(monkeypatch):
    monkeypatch.setenv(demo_scenarios.DEMO_ENV, "1")
    monkeypatch.setattr(demo_scenarios, "SPEED", 0)
    agent, _ = make_agent()
    agent.issue_analyzer.client = type("C", (), {"responses": ExplodingAnalyzerResponses()})()

    result = agent.understand(contract=CONTRACT, message=QUICK, demo=True)

    assert result["inScope"] is True
    assert any(question["id"] == "partial_payment" for question in result["questions"])


def test_demo_write_streams_like_real(monkeypatch):
    monkeypatch.setenv(demo_scenarios.DEMO_ENV, "1")
    monkeypatch.setattr(demo_scenarios, "SPEED", 0)
    agent, _ = make_agent()
    events = []

    result = agent.suggest_clause(
        contract=CONTRACT,
        message=QUICK,
        issues=[{"type": "DEFAULT"}, {"type": "ACCELERATION"}],
        slots=[],
        questions=[],
        answers={},
        demo=True,
        on_event=lambda event, data: events.append(event),
    )

    assert events[0] == "progress"
    assert "references" in events and "delta" in events and "draft" in events
    assert result["suggestedClause"].startswith("채무자가 각 회차 상환기일까지")