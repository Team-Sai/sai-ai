"""
특약 생성 스트리밍(SSE) 테스트.

실제 OpenAI / Qdrant / 모델 없이
이벤트 순서, 최종 응답, 오류 처리, keep-alive를 확인한다.
"""

import importlib.util
import json
import os
import sys
import time
import types

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

# law_retriever는 qdrant_client와 임베딩 모델을 불러오므로 가짜 모듈로 대체
if "sai_rag.retrieval.law_retriever" not in sys.modules:
    fake_retriever_module = types.ModuleType("sai_rag.retrieval.law_retriever")

    class FakeLawRetriever:  # noqa: D401 - 타입 자리만 채움
        pass

    fake_retriever_module.LawRetriever = FakeLawRetriever
    sys.modules["sai_rag.retrieval.law_retriever"] = fake_retriever_module

# contract_review_service는 "src." 경로로 같은 모듈을 불러온다
if importlib.util.find_spec("qdrant_client") is None:
    sys.modules.setdefault(
        "src.sai_rag.retrieval.law_retriever",
        sys.modules["sai_rag.retrieval.law_retriever"],
    )

# torch가 없는 환경(로컬 파이썬 등)에서도 API 테스트가 돌도록 리랭커 팩토리를 가짜로 대체
# (Docker 이미지에는 torch가 있으므로 그대로 진짜를 쓴다)
if importlib.util.find_spec("torch") is None and "sai_rag.reranking.factory" not in sys.modules:
    fake_reranker_factory = types.ModuleType("sai_rag.reranking.factory")
    fake_reranker_factory.create_reranker = lambda *args, **kwargs: None
    sys.modules["sai_rag.reranking.factory"] = fake_reranker_factory

from fastapi.testclient import TestClient  # noqa: E402

from sai_rag.api.clause_stream import (  # noqa: E402
    format_sse,
    stream_clause_events,
    to_agent_contract,
)

CONTRACT = {"principal": 1_000_000, "repaymentDate": "2027-01-31"}
DRAFT = "제1조(중도상환) 채무자는 2027년 1월 30일 전 언제든 상환할 수 있다."
FINAL = "제1조(중도상환) 채무자는 2027년 1월 31일 전 언제든 상환할 수 있다."


def parse_sse(chunks):
    """SSE 문자열 조각들을 (event, data) 목록으로 바꾼다. keep-alive 주석은 따로 센다."""
    events, comments = [], 0
    for block in "".join(chunks).split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            comments += 1
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events, comments


class FakeAgent:
    def __init__(self, final_clause=DRAFT, error=None, delay=0.0):
        self.final_clause = final_clause
        self.error = error
        self.delay = delay
        self.received = None

    def suggest_clause(self, *, contract, message, on_event=None):
        self.received = (contract, message)
        on_event = on_event or (lambda event, data: None)
        on_event("progress", {"step": "analysis"})
        if self.error:
            raise self.error
        on_event("progress", {"step": "retrieval"})
        on_event("progress", {"step": "generation"})
        on_event("draft", {"title": "중도상환", "suggestedClause": DRAFT})
        on_event("progress", {"step": "validation"})
        time.sleep(self.delay)
        return {
            "suggestedClause": self.final_clause,
            "validation": {"status": "PASS", "isSafeToSuggest": True},
            "debug": {"searchPlans": ["내부 검색 질의"], "llmReasons": []},
        }


def run_stream(agent, **kwargs):
    return parse_sse(
        list(
            stream_clause_events(
                clause_agent=agent,
                contract=CONTRACT,
                message="중도상환 수수료 없이",
                **kwargs,
            )
        )
    )


# ---------------------------------------------------------------
# 1. 스트림 도우미
# ---------------------------------------------------------------

def test_events_come_in_order_and_end_with_final():
    events, _ = run_stream(FakeAgent())

    names = [e for e, _ in events]
    assert names == ["progress", "progress", "progress", "draft", "progress", "final"]
    assert [d["step"] for e, d in events if e == "progress"] == [
        "analysis",
        "retrieval",
        "generation",
        "validation",
    ]


def test_final_has_clause_revised_false_when_unchanged():
    events, _ = run_stream(FakeAgent(final_clause=DRAFT))
    final = events[-1][1]
    assert final["clauseRevised"] is False
    assert final["validation"]["status"] == "PASS"


def test_final_has_clause_revised_true_when_validator_changed_clause():
    events, _ = run_stream(FakeAgent(final_clause=FINAL))
    assert events[-1][1]["clauseRevised"] is True


def test_final_does_not_include_debug():
    events, _ = run_stream(FakeAgent())
    final = events[-1][1]
    assert "debug" not in final
    assert final["suggestedClause"] == DRAFT


def test_value_error_becomes_400_error_event():
    events, _ = run_stream(FakeAgent(error=ValueError("특약 요청 내용을 입력해주세요.")))
    assert events[-1] == ("error", {"status": 400, "message": "특약 요청 내용을 입력해주세요."})


def test_unexpected_error_becomes_500_error_event():
    events, _ = run_stream(FakeAgent(error=RuntimeError("boom")))
    event, data = events[-1]
    assert event == "error"
    assert data["status"] == 500
    assert "boom" not in data["message"]  # 내부 오류 내용은 밖으로 내보내지 않는다


def test_keep_alive_is_sent_while_waiting():
    events, comments = run_stream(FakeAgent(delay=0.35), heartbeat_seconds=0.1)
    assert comments >= 2
    assert events[-1][0] == "final"


def test_format_sse_keeps_korean():
    assert format_sse("progress", {"step": "분석"}) == 'event: progress\ndata: {"step": "분석"}\n\n'


def test_to_agent_contract_adds_aliases_without_mutating_input():
    original = dict(CONTRACT)
    converted = to_agent_contract(original)
    assert converted["principalAmount"] == 1_000_000
    assert converted["maturityDate"] == "2027-01-31"
    assert "principalAmount" not in original


# ---------------------------------------------------------------
# 2. ContractClauseAgent: 범위 밖 요청은 분석 이벤트만 보내고 끝난다
# ---------------------------------------------------------------

def test_agent_out_of_scope_emits_only_analysis_progress():
    from sai_rag.agent.contract_clause_agent import ContractClauseAgent

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"범위 밖인데 {name} 호출")

    class FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text=json.dumps({"inScope": False, "issues": []}))

    agent = ContractClauseAgent(retriever=Exploding(), generator=Exploding(), reranker=Exploding())
    agent.issue_analyzer.client = types.SimpleNamespace(responses=FakeResponses())

    received = []
    result = agent.suggest_clause(
        contract=CONTRACT,
        message="김치찌개 만드는 법",
        on_event=lambda e, d: received.append((e, d)),
    )

    assert received == [("progress", {"step": "analysis"})]
    assert result["validation"]["status"] == "OUT_OF_SCOPE"


def test_agent_keeps_working_when_event_callback_fails():
    from sai_rag.agent.contract_clause_agent import ContractClauseAgent

    def broken(event, data):
        raise RuntimeError("client gone")

    ContractClauseAgent._emit(broken, "progress", {"step": "analysis"})  # 예외가 밖으로 나오면 실패


# ---------------------------------------------------------------
# 3. API 엔드포인트
# ---------------------------------------------------------------

def test_stream_endpoint_returns_event_stream():
    from sai_rag.api import main

    agent = FakeAgent(final_clause=FINAL)
    main.app.state.clause_agent = agent
    client = TestClient(main.app)  # with 없이 만들면 lifespan(모델 로딩)을 실행하지 않는다

    response = client.post(
        "/rag/contracts/clause-suggestions/stream",
        json={"contract": CONTRACT, "message": "중도상환 수수료 없이"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events, _ = parse_sse([response.text])
    assert events[0] == ("progress", {"step": "analysis"})
    assert events[-1][0] == "final"
    assert events[-1][1]["clauseRevised"] is True
    # 기존 API와 같은 alias 변환을 거친다
    assert agent.received[0]["principalAmount"] == 1_000_000


def test_existing_endpoint_still_returns_json():
    from sai_rag.api import main

    main.app.state.clause_agent = FakeAgent()
    client = TestClient(main.app)

    response = client.post(
        "/rag/contracts/clause-suggestions",
        json={"contract": CONTRACT, "message": "중도상환 수수료 없이"},
    )

    assert response.status_code == 200
    assert response.json()["suggestedClause"] == DRAFT