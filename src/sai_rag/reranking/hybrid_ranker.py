from typing import Any


class HybridRanker:
    """
    Vector Search 점수와 Cross-Encoder Reranker 점수를
    함께 사용하여 최종 순위를 계산한다.

    두 점수는 서로 단위가 다르므로
    각각 0~1 범위로 Min-Max 정규화한 뒤 결합한다.

    기본 가중치:
    - Reranker: 0.7
    - Vector Search: 0.3

    Reranker를 주 판단 기준으로 사용하되,
    Reranker 점수가 비슷한 후보 사이에서는
    Vector Search의 의미 유사도를 보조 신호로 사용한다.
    """

    def __init__(
        self,
        rerank_weight: float = 0.7,
        vector_weight: float = 0.3,
    ):
        total = (
            rerank_weight
            + vector_weight
        )

        if total <= 0:
            raise ValueError(
                "Ranking weight의 합은 0보다 커야 합니다."
            )

        self.rerank_weight = (
            rerank_weight / total
        )

        self.vector_weight = (
            vector_weight / total
        )

    def rank(
        self,
        candidates: list[
            dict[str, Any]
        ],
        top_k: int = 5,
    ) -> list[
        dict[str, Any]
    ]:
        if not candidates:
            return []

        vector_scores = [
            float(
                candidate.get(
                    "score",
                    0,
                )
            )
            for candidate
            in candidates
        ]

        rerank_scores = [
            float(
                candidate.get(
                    "rerank_score",
                    0,
                )
            )
            for candidate
            in candidates
        ]

        normalized_vectors = (
            self._min_max_normalize(
                vector_scores
            )
        )

        normalized_reranks = (
            self._min_max_normalize(
                rerank_scores
            )
        )

        ranked: list[
            dict[str, Any]
        ] = []

        for (
            candidate,
            vector_normalized,
            rerank_normalized,
        ) in zip(
            candidates,
            normalized_vectors,
            normalized_reranks,
        ):
            final_score = (
                self.rerank_weight
                * rerank_normalized
                +
                self.vector_weight
                * vector_normalized
            )

            ranked.append(
                {
                    **candidate,

                    "vector_normalized": (
                        vector_normalized
                    ),

                    "rerank_normalized": (
                        rerank_normalized
                    ),

                    "final_score": (
                        final_score
                    ),
                }
            )

        ranked.sort(
            key=lambda item: float(
                item.get(
                    "final_score",
                    0,
                )
            ),
            reverse=True,
        )

        return ranked[
            :top_k
        ]

    def _min_max_normalize(
        self,
        values: list[float],
    ) -> list[float]:
        if not values:
            return []

        minimum = min(
            values
        )

        maximum = max(
            values
        )

        # 모든 점수가 동일한 경우
        # 특정 후보를 임의로 우대하지 않는다.
        if maximum == minimum:
            return [
                1.0
                for _ in values
            ]

        return [
            (
                value - minimum
            )
            /
            (
                maximum - minimum
            )
            for value in values
        ]