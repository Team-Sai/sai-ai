from typing import Any

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from sai_rag.reranking.base import (
    RerankerProvider,
)


class BgeReranker(
    RerankerProvider
):
    """
    BAAI BGE Reranker 기반
    Cross-Encoder Reranker.

    Vector Search와 달리
    query와 document를 하나의 입력 쌍으로 모델에 전달하여
    관련성을 다시 계산한다.
    """

    def __init__(
        self,
        model_name: str = (
            "BAAI/bge-reranker-v2-m3"
        ),
        batch_size: int = 4,
    ):
        self._model_name = (
            model_name
        )

        self.batch_size = (
            batch_size
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "[BgeReranker] "
            f"device={self.device}"
        )

        print(
            "[BgeReranker] "
            f"model={self._model_name}"
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self._model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self._model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    @property
    def name(self) -> str:
        return "bge-reranker"

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
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

        scored_candidates: list[
            dict[str, Any]
        ] = []

        for start in range(
            0,
            len(candidates),
            self.batch_size,
        ):
            batch = candidates[
                start:
                start + self.batch_size
            ]

            queries: list[str] = []

            documents: list[str] = []

            for candidate in batch:
                rerank_query = str(
                    candidate.get(
                        "_rerank_query",
                        candidate.get(
                            "_query",
                            "",
                        ),
                    )
                )

                document_text = str(
                    candidate.get(
                        "text",
                        "",
                    )
                )

                queries.append(
                    rerank_query
                )

                documents.append(
                    document_text
                )

            inputs = (
                self.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value
                in inputs.items()
            }

            with torch.no_grad():
                outputs = (
                    self.model(
                        **inputs
                    )
                )

            logits = (
                outputs.logits
            )

            # 대부분의 reranker는
            # 문서당 하나의 relevance logit을 반환한다.
            if (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):
                scores = (
                    logits[:, 0]
                )

            # 혹시 2-class 구조라면
            # 마지막 class를 positive relevance로 사용
            elif (
                logits.ndim == 2
                and logits.shape[1] > 1
            ):
                scores = (
                    logits[:, -1]
                )

            else:
                scores = (
                    logits.reshape(-1)
                )

            scores = (
                scores
                .detach()
                .cpu()
                .tolist()
            )

            for (
                candidate,
                score,
            ) in zip(
                batch,
                scores,
            ):
                result = {
                    **candidate,
                    "rerank_score": (
                        float(score)
                    ),
                }

                scored_candidates.append(
                    result
                )

        scored_candidates.sort(
            key=lambda item: float(
                item.get(
                    "rerank_score",
                    0,
                )
            ),
            reverse=True,
        )

        return scored_candidates[
            :top_k
        ]