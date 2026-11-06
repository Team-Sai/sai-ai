from contextlib import (
    asynccontextmanager,
)
import os

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import (
    StreamingResponse,
)

from sai_rag.agent.clause_generator import (
    ClauseGenerator,
)

from sai_rag.agent.chat_router import (
    route_reply,
)

from sai_rag.agent.contract_clause_agent import (
    ContractClauseAgent,
)

from sai_rag.api.schemas import (
    ClauseChatPrepareRequest,
    ClauseChatRouteRequest,
    ClauseChatWriteRequest,
    ClauseReviewRequest,
    ContractClauseSuggestionRequest,
    ContractReviewRequest,
)

from sai_rag.api.warmup import warm_up
from sai_rag.api.clause_stream import (
    stream_clause_events,
    to_agent_contract,
)

from sai_rag.generation.factory import (
    create_review_generator,
)

from sai_rag.reranking.factory import (
    create_reranker,
)

from sai_rag.retrieval.law_retriever import (
    LawRetriever,
)

from sai_rag.review.contract_review_service import (
    ContractReviewService,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    print()

    print(
        "=" * 60
    )

    print(
        "[Startup] "
        "SAI AI Service 초기화"
    )

    print(
        "=" * 60
    )

    # =========================================
    # Retrieval
    # =========================================

    qdrant_url = os.getenv(
        "QDRANT_URL"
    )

    if not qdrant_url:
        raise RuntimeError(
            "QDRANT_URL 환경변수가 필요합니다."
        )

    retriever = (
        LawRetriever(
            provider="bge-m3",
            qdrant_url=qdrant_url,
            collection_name=os.getenv(
                "QDRANT_COLLECTION",
                "law_chunks_bge_m3",
            ),
        )
    )

    # =========================================
    # 기존 계약 검토 Generator
    # =========================================

    review_generator = (
        create_review_generator(
            "openai"
        )
    )

    review_service = (
        ContractReviewService(
            retriever=retriever,
            generator=(
                review_generator
            ),
        )
    )

    # =========================================
    # Clause Agent Reranker
    # =========================================

    reranker = (
        create_reranker(
            "bge-reranker"
        )
    )

    # =========================================
    # Clause Agent Generator
    # =========================================

    clause_generator = (
        ClauseGenerator()
    )

    clause_agent = (
        ContractClauseAgent(
            retriever=retriever,
            generator=(
                clause_generator
            ),
            reranker=reranker,
        )
    )

    # =========================================
    # 준비 운동: 첫 사용자 요청이 느리지 않게 검색·순위 모델을 미리 돌림
    # (OpenAI는 부르지 않아 비용 없음)
    # =========================================

    warm_up(
        retriever=retriever,
        reranker=reranker,
    )

    app.state.review_service = (
        review_service
    )

    app.state.clause_agent = (
        clause_agent
    )

    print()

    print(
        "[Startup] "
        "Contract Review Service 준비 완료"
    )

    print(
        "[Startup] "
        "Contract Clause Agent 준비 완료"
    )

    print(
        "[Startup] "
        "BGE Reranker 준비 완료"
    )

    print()

    yield

    print()

    print(
        "[Shutdown] "
        "SAI AI Service 종료"
    )


app = FastAPI(
    title="Saiwonjang AI API",

    description=(
        "사이원장 차용증 법률 검토 및 "
        "AI 특약 작성 Agent"
    ),

    version="0.3.0",

    lifespan=lifespan,
)


@app.get(
    "/health"
)
def health():
    return {
        "status": "ok",

        "services": {
            "contractReview": (
                "ready"
            ),

            "clauseAgent": (
                "ready"
            ),

            "reranker": (
                "ready"
            ),
        },
    }

@app.post(
    "/rag/contracts/review"
)
def review_contract(
    request: Request,
    body: ContractReviewRequest,
):
    try:
        review_service = (
            request.app.state
            .review_service
        )

        contract = (
            body.model_dump()
        )

        return (
            review_service.review(
                contract=contract,
                top_k=3,
                generate=True,
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            "[Review Error]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "차용증 법률 검토 중 "
                "오류가 발생했습니다."
            ),
        ) from exc

@app.post(
    "/rag/contracts/clause-suggestions"
)
def suggest_contract_clause(
    request: Request,
    body: ContractClauseSuggestionRequest,
):
    try:
        clause_agent = (
            request.app.state
            .clause_agent
        )

        contract = to_agent_contract(
            body.contract.model_dump()
        )

        # 후보 수 / 최종 근거 수는
        # ContractClauseAgent 내부 정책을 사용한다.

        return (
            clause_agent.suggest_clause(
                contract=contract,
                message=(
                    body.message
                ),
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            "[Clause Agent Error]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI 특약 생성 중 "
                "오류가 발생했습니다."
            ),
        ) from exc


@app.post(
    "/rag/contracts/clause-chat/understand"
)
def understand_contract_clause(
    request: Request,
    body: ContractClauseSuggestionRequest,
):
    """
    대화형 특약 1턴: 쓰기 전에 이해한 내용과 정해야 할 것을 돌려준다.
    쟁점 분석(LLM 1번)만 하고 검색·생성은 하지 않는다.
    """

    try:
        clause_agent = (
            request.app.state
            .clause_agent
        )

        contract = to_agent_contract(
            body.contract.model_dump()
        )

        return clause_agent.understand(
            contract=contract,
            message=body.message,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            "[Clause Understand Error]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "요청 내용을 확인하는 중 "
                "오류가 발생했습니다."
            ),
        ) from exc


@app.post(
    "/rag/contracts/clause-chat/route"
)
def route_contract_clause_reply(
    body: ClauseChatRouteRequest,
):
    """
    대화형 특약: 사용자 답장을 규칙으로 분류한다. (LLM 0번)

    확정 / 거절 / 값 바꾸기 / 질문 답 / 모름 / 애매 중 하나와
    고친 값 칸·답 목록을 돌려준다. 애매하면 아무것도 바꾸지 않는다.
    """

    try:
        result = route_reply(
            message=body.message,
            contract=to_agent_contract(
                body.contract.model_dump()
            ),
            slots=body.slots,
            questions=body.questions,
            answers=body.answers,
        )

    except (KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "대화 상태 형식이 "
                "올바르지 않습니다."
            ),
        ) from exc

    print(
        "[CHAT] "
        f"kind={result['kind']} "
        f"route={'UNDECIDED' if result['kind'] == 'AMBIGUOUS' else 'RULE'} "
        f"message_chars={len(body.message.strip())} "
        f"pending={len(result['pendingQuestions'])}"
    )

    return result


@app.post(
    "/rag/contracts/clause-chat/prepare"
)
def prepare_contract_clause(
    request: Request,
    body: ClauseChatPrepareRequest,
):
    """
    대화형 특약: 사용자가 답하는 동안 ② 검색 + ③ 순위를 미리 한다.
    백엔드가 결과를 보관했다가 /write/stream에 references로 같이 보낸다.
    """

    try:
        references = request.app.state.clause_agent.prepare_references(
            contract=to_agent_contract(
                body.contract.model_dump()
            ),
            message=body.message,
            issues=body.issues,
        )

    except Exception as exc:
        print(
            "[Clause Prepare Error]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="근거 법령을 미리 찾는 중 오류가 발생했습니다.",
        ) from exc

    print(f"[CHAT] prepare references={len(references)}")

    return {"references": references}


@app.post(
    "/rag/contracts/clause-review"
)
def review_contract_clause(
    request: Request,
    body: ClauseReviewRequest,
):
    """
    반영한 AI 조항을 다시 검토한다. (생성 없이 ⑤ 검증만)
    규칙으로 끝나면 LLM을 부르지 않고, 필요할 때만 LLM 검증을 한다.
    """

    try:
        return request.app.state.clause_agent.review_clause(
            contract=to_agent_contract(
                body.contract.model_dump()
            ),
            message=body.message,
            issues=body.issues,
            clause=body.clause,
            references=body.references,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            "[Clause Review Error]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail="조항을 다시 검토하는 중 오류가 발생했습니다.",
        ) from exc


@app.post(
    "/rag/contracts/clause-suggestions/stream"
)
def suggest_contract_clause_stream(
    request: Request,
    body: ContractClauseSuggestionRequest,
):
    """
    특약 생성 진행 상황을 SSE로 보낸다.

    progress → draft(생성 직후 초안) → final(기존 응답과 같은 모양 + clauseRevised)
    오류가 나면 error 이벤트를 보낸다.
    """

    clause_agent = (
        request.app.state
        .clause_agent
    )

    contract = to_agent_contract(
        body.contract.model_dump()
    )

    return StreamingResponse(
        stream_clause_events(
            clause_agent=clause_agent,
            contract=contract,
            message=body.message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post(
    "/rag/contracts/clause-chat/write/stream"
)
def write_contract_clause_stream(
    request: Request,
    body: ClauseChatWriteRequest,

):
    """
    대화형 특약 2턴: 확정한 조건으로 특약을 쓴다. (SSE)

    /understand에서 받은 쟁점을 그대로 써서 ① 쟁점 분석을 다시 하지 않고,
    ② 검색 → ③ 순위 → ④ 생성 → ⑤ 검증을 한다.
    이벤트는 /clause-suggestions/stream과 같다. (progress는 retrieval부터)
    """

    clause_agent = (
        request.app.state
        .clause_agent
    )

    contract = to_agent_contract(
        body.contract.model_dump()
    )

    return StreamingResponse(
        stream_clause_events(
            clause_agent=clause_agent,
            contract=contract,
            message=body.message,
            agent_options={
                "issues": body.issues,
                "slots": body.slots,
                "questions": body.questions,
                "answers": body.answers,
                "references": body.references,
            },
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

