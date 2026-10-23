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
import pytest

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

    def suggest_clause(self, *, contract, message, on_event=None, should_stop=None):
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


# ---------------------------------------------------------------
# API 경로 회귀 테스트: 백엔드(RagApiClient)가 부르는 경로가
# 빠지거나 두 번 등록되지 않았는지 확인한다.
# ---------------------------------------------------------------

def _post_paths():
    from sai_rag.api import main

    return [
        route.path
        for route in main.app.routes
        if "POST" in getattr(route, "methods", set())
    ]


def test_backend_routes_exist():
    assert {
        "/rag/contracts/review",
        "/rag/contracts/clause-suggestions",
        "/rag/contracts/clause-suggestions/stream",
    } <= set(_post_paths())


def test_no_duplicate_post_routes():
    paths = _post_paths()
    assert len(paths) == len(set(paths))


# ---------------------------------------------------------------
# 2차: 조항 글자 스트리밍(delta)과 취소
# ---------------------------------------------------------------

class DeltaAgent:
    """생성 중 조항 글자를 여러 조각으로 보내는 가짜 Agent."""

    def suggest_clause(self, *, contract, message, on_event=None, should_stop=None):
        on_event("progress", {"step": "generation"})
        for piece in ["제1조 ", "채무자는 ", "언제든지 ", "상환할 수 있다."]:
            on_event("delta", {"text": piece})
        on_event("draft", {"title": "중도상환", "suggestedClause": DRAFT})
        return {"suggestedClause": DRAFT, "validation": {"status": "PASS"}}


def test_delta_events_are_merged_and_come_before_draft():
    events, _ = run_stream(DeltaAgent())

    names = [e for e, _ in events]
    deltas = "".join(d["text"] for e, d in events if e == "delta")

    assert deltas == "제1조 채무자는 언제든지 상환할 수 있다."
    # 조각이 한꺼번에 쌓였으면 하나로 합쳐 보낸다 (최대 4개)
    assert 1 <= names.count("delta") <= 4
    assert names.index("draft") > max(i for i, n in enumerate(names) if n == "delta")
    assert names[-1] == "final"


class WaitingAgent:
    """should_stop이 True가 될 때까지 기다리는 가짜 Agent."""

    def __init__(self):
        import threading

        self.stopped = threading.Event()

    def suggest_clause(self, *, contract, message, on_event=None, should_stop=None):
        from sai_rag.agent.clause_generator import ClauseGenerationCancelled

        on_event("progress", {"step": "analysis"})
        deadline = time.time() + 5
        while time.time() < deadline:
            if should_stop():
                self.stopped.set()
                raise ClauseGenerationCancelled()
            time.sleep(0.01)
        return {"suggestedClause": DRAFT}


def test_closing_stream_tells_agent_to_stop():
    agent = WaitingAgent()
    stream = stream_clause_events(
        clause_agent=agent,
        contract=CONTRACT,
        message="중도상환",
        heartbeat_seconds=0.05,
    )

    assert next(stream).startswith("event: progress")

    stream.close()  # 브라우저/백엔드가 연결을 끊은 상황

    assert agent.stopped.wait(timeout=2)


def _generator_with_events(events):
    from sai_rag.agent.clause_generator import ClauseGenerator

    class FakeStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            return iter(events)

        def close(self):
            self.closed = True

    fake_stream = FakeStream()
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return fake_stream

    generator = ClauseGenerator(api_key="test-key-not-real")
    generator.client = types.SimpleNamespace(responses=FakeResponses())
    return generator, fake_stream, calls


def _delta(text):
    return types.SimpleNamespace(type="response.output_text.delta", delta=text)


def test_generator_streams_clause_text_while_writing_json():
    full = json.dumps(
        {
            "suggestedClause": "제1조 채무자는\n언제든지 \"상환\"할 수 있다.",
            "title": "중도상환",
            "summary": "요약",
            "warnings": [],
            "unresolvedDecisions": [],
        },
        ensure_ascii=False,
    )
    # 이스케이프 중간에서 잘리도록 3글자씩 나눈다
    events = [_delta(full[i:i + 3]) for i in range(0, len(full), 3)]
    generator, fake_stream, calls = _generator_with_events(events)

    received = []
    result = generator.generate(
        message="중도상환 자유롭게",
        contract=CONTRACT,
        issues=[{"type": "EARLY_REPAYMENT"}],
        references=[],
        on_delta=received.append,
    )

    assert calls[0]["stream"] is True
    assert "".join(received) == "제1조 채무자는\n언제든지 \"상환\"할 수 있다."
    assert result["suggestedClause"].startswith("제1조 채무자는")
    assert fake_stream.closed


def test_generator_stops_when_cancelled():
    from sai_rag.agent.clause_generator import ClauseGenerationCancelled

    events = [_delta('{"suggestedClause": "제1조'), _delta(" 채무자는")]
    generator, fake_stream, _ = _generator_with_events(events)

    with pytest.raises(ClauseGenerationCancelled):
        generator.generate(
            message="중도상환",
            contract=CONTRACT,
            issues=[],
            references=[],
            on_delta=lambda text: None,
            should_stop=lambda: True,
        )

    assert fake_stream.closed  # 모델 응답 연결을 닫아 토큰을 더 쓰지 않는다
