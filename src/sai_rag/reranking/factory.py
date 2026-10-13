from sai_rag.reranking.base import (
    RerankerProvider,
)

from sai_rag.reranking.bge_reranker import (
    BgeReranker,
)


def create_reranker(
    provider: str = (
        "bge-reranker"
    ),
) -> RerankerProvider:
    normalized = (
        provider
        .strip()
        .lower()
    )

    if normalized in {
        "bge",
        "bge-reranker",
        "bge-reranker-v2-m3",
    }:
        return (
            BgeReranker(
                batch_size=8,
                max_length=512,
            )
        )

    raise ValueError(
        "지원하지 않는 "
        "Reranker provider입니다: "
        f"{provider}"
    )