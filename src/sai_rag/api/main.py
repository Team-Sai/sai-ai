from contextlib import (
    asynccontextmanager,
)
import os

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
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

        contract = (
            body.contract.model_dump()
        )

        # API 요청 모델은 기존 호환성을 위해
        # principal / repaymentDate를 유지한다.
        #
        # Agent/Generator에서는 의미가 더 명확한
        # principalAmount / maturityDate도 함께 사용하도록
        # canonical alias를 추가한다.
        #
        # 기존 ContractDomainValidator는 principal /
        # repaymentDate를 계속 읽기 때문에 원래 key도 보존한다.
        if (
            contract.get(
                "principalAmount"
            )
            is None
        ):
            contract[
                "principalAmount"
            ] = contract.get(
                "principal"
            )

        if (
            contract.get(
                "maturityDate"
            )
            is None
        ):
            contract[
                "maturityDate"
            ] = contract.get(
                "repaymentDate"
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
