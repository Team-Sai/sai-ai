from typing import Any

from qdrant_client import (
    QdrantClient,
)
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
)

from sai_rag.embedding.factory import (
    create_embedder,
)


class LawRetriever:
    """
    Qdrant 기반 법령 Retriever.

    기본 Semantic Search와
    법령명 metadata filtering을 지원한다.
    """

    def __init__(
        self,
        provider: str = "bge-m3",
        qdrant_url: str = (
            "http://localhost:6333"
        ),
        collection_name: str = (
            "law_chunks_bge_m3"
        ),
    ):
        self.embedder = (
            create_embedder(
                provider
            )
        )

        self.client = (
            QdrantClient(
                url=qdrant_url
            )
        )

        self.collection_name = (
            collection_name
        )

        if not self.client.collection_exists(
            collection_name=(
                self.collection_name
            )
        ):
            raise ValueError(
                "Qdrant collection을 찾을 수 없습니다: "
                f"{self.collection_name}"
            )

        print(
            "[LawRetriever] "
            f"provider={provider}"
        )

        print(
            "[LawRetriever] "
            f"collection={self.collection_name}"
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        law_names: (
            list[str] | None
        ) = None,
    ) -> list[
        dict[str, Any]
    ]:
        if not question.strip():
            raise ValueError(
                "검색 질문이 비어 있습니다."
            )

        query_vector = (
            self.embedder.embed_query(
                question
            )
        )

        query_filter = (
            self._build_law_filter(
                law_names
            )
        )

        response = (
            self.client.query_points(
                collection_name=(
                    self.collection_name
                ),
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        )

        results: list[
            dict[str, Any]
        ] = []

        for point in response.points:
            payload = (
                point.payload or {}
            )

            result = {
                "score": float(
                    point.score
                ),
                "chunk_id": (
                    payload.get(
                        "chunk_id"
                    )
                ),
                "document_id": (
                    payload.get(
                        "document_id"
                    )
                ),
                "text": (
                    payload.get(
                        "text",
                        "",
                    )
                ),
                "metadata": {
                    "source_type": (
                        payload.get(
                            "source_type"
                        )
                    ),
                    "document_kind": (
                        payload.get(
                            "document_kind"
                        )
                    ),
                    "law_id": (
                        payload.get(
                            "law_id"
                        )
                    ),
                    "law_name": (
                        payload.get(
                            "law_name"
                        )
                    ),
                    "article_number": (
                        payload.get(
                            "article_number"
                        )
                    ),
                    "article_title": (
                        payload.get(
                            "article_title"
                        )
                    ),
                    "effective_date": (
                        payload.get(
                            "effective_date"
                        )
                    ),
                    "promulgation_date": (
                        payload.get(
                            "promulgation_date"
                        )
                    ),
                    "revision_type": (
                        payload.get(
                            "revision_type"
                        )
                    ),
                    "source": (
                        payload.get(
                            "source"
                        )
                    ),
                },
            }

            results.append(
                result
            )

        return results

    def _build_law_filter(
        self,
        law_names: (
            list[str] | None
        ),
    ) -> Filter | None:
        if not law_names:
            return None

        normalized_names = [
            name.strip()
            for name in law_names
            if name.strip()
        ]

        if not normalized_names:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="law_name",
                    match=MatchAny(
                        any=normalized_names
                    ),
                )
            ]
        )