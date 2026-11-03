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

from sai_rag.agent.contract_clause_agent import (
    ContractClauseAgent,
)

from sai_rag.api.schemas import (
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