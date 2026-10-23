"""
AI 특약 생성 스트리밍(SSE) 도우미.

특약 생성은 분석 → 검색 → 생성 → 검증 순서로 15~35초가 걸린다.
기존 API는 모두 끝난 뒤 한 번에 응답하므로, 사용자는 그동안 진행 상황을 알 수 없다.

여기서는 같은 ContractClauseAgent.suggest_clause()를 별도 스레드에서 실행하고,
단계마다 알려주는 이벤트를 SSE(Server-Sent Events) 형식으로 흘려보낸다.

    event: progress   data: {"step": "analysis"}          단계 시작
    event: references data: {"references": [...]}          생성에 쓰는 근거 법령 (생성 직전)
    event: delta      data: {"text": "새로 써진 조항 글자"}  생성 모델이 조항을 쓰는 중
    event: draft      data: {..., "suggestedClause": ...} 생성 직후 초안 (검증 전)
    event: final      data: {기존 응답과 같은 모양, "clauseRevised": bool}
    event: error      data: {"status": 400 | 500, "message": "..."}

연결이 오래 조용하면 프록시/로드밸런서가 끊을 수 있어서
일정 시간마다 SSE 주석(": keep-alive")을 보낸다.

브라우저(또는 백엔드)가 연결을 끊으면 이 제너레이터가 닫히고,
작업 스레드에 "그만" 신호를 보내 다음 단계로 넘어가지 않게 한다.
(생성 중이면 모델 응답 받기를 멈춰서 토큰도 더 쓰지 않는다)
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator

from sai_rag.agent.clause_generator import ClauseGenerationCancelled

_DONE = object()


def to_agent_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """
    API 요청 모델은 기존 호환성을 위해 principal / repaymentDate를 유지한다.
    Agent/Generator는 의미가 더 명확한 principalAmount / maturityDate도 함께 쓰므로
    canonical alias를 추가한다. (원래 key도 보존)
    """

    contract = dict(contract)

    if contract.get("principalAmount") is None:
        contract["principalAmount"] = contract.get("principal")

    if contract.get("maturityDate") is None:
        contract["maturityDate"] = contract.get("repaymentDate")

    return contract


def format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def stream_clause_events(
    *,
    clause_agent: Any,
    contract: dict[str, Any],
    message: str,
    heartbeat_seconds: float = 15.0,
) -> Iterator[str]:
    """
    suggest_clause()를 백그라운드 스레드에서 실행하고 이벤트를 SSE 문자열로 내보낸다.
    """

    events: queue.Queue = queue.Queue()
    cancelled = threading.Event()

    def on_event(event: str, data: dict[str, Any]) -> None:
        events.put((event, data))

    def run() -> None:
        try:
            result = clause_agent.suggest_clause(
                contract=contract,
                message=message,
                on_event=on_event,
                should_stop=cancelled.is_set,
            )
            events.put(("final", result))
        except ClauseGenerationCancelled:
            # 받을 사람이 없으므로 이벤트를 보내지 않고 끝낸다.
            print("[Clause Agent Stream] 클라이언트 연결 종료로 생성 중단")
        except ValueError as exc:
            events.put(("error", {"status": 400, "message": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - 어떤 오류든 error 이벤트로 알린다
            print("[Clause Agent Stream Error]", repr(exc))
            events.put(
                (
                    "error",
                    {"status": 500, "message": "AI 특약 생성 중 오류가 발생했습니다."},
                )
            )
        finally:
            events.put(_DONE)

    threading.Thread(target=run, name="clause-stream", daemon=True).start()

    try:
        yield from _relay_events(events, heartbeat_seconds)
    finally:
        # 정상 종료든 연결 끊김(제너레이터 close)이든 작업 스레드에 그만하라고 알린다.
        cancelled.set()


def _relay_events(
    events: queue.Queue,
    heartbeat_seconds: float,
) -> Iterator[str]:
    draft_clause: str | None = None
    pending: list[Any] = []

    while True:
        if pending:
            item = pending.pop(0)
        else:
            try:
                item = events.get(timeout=heartbeat_seconds)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

        if item is _DONE:
            return

        event, data = item

        if event == "delta":
            # 글자 조각이 쌓여 있으면 한 이벤트로 합쳐 보낸다. (이벤트 수 줄이기)
            text = data.get("text") or ""

            while True:
                try:
                    following = events.get_nowait()
                except queue.Empty:
                    break

                if following is not _DONE and following[0] == "delta":
                    text += following[1].get("text") or ""
                    continue

                pending.append(following)
                break

            if text:
                yield format_sse("delta", {"text": text})

            continue

        if event == "draft":
            draft_clause = (data.get("suggestedClause") or "").strip()

        if event == "final":
            # debug(검색 질의, 후보 점수, 내부 검증 사유 등)는 내부 확인용이다.
            # 기존 API는 백엔드 DTO가 이 필드를 버려서 브라우저까지 가지 않았지만,
            # 스트리밍은 백엔드가 그대로 중계하므로 여기서 뺀다.
            data = {key: value for key, value in data.items() if key != "debug"}
            final_clause = (data.get("suggestedClause") or "").strip()
            # 검증에서 문장이 바뀌었는지 프론트가 직접 비교하지 않아도 되도록 알려준다.
            data["clauseRevised"] = bool(
                draft_clause and final_clause and draft_clause != final_clause
            )

        yield format_sse(event, data)