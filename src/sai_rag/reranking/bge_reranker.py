from __future__ import annotations

from time import perf_counter
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
    BAAI/bge-reranker-v2-m3 기반 Reranker.

    V3

    - CUDA 우선
    - batch_size=8
    - max_length=512
    - inference_mode
    - CUDA autocast
    - batch별 성능 측정
    """

    DEFAULT_MODEL_NAME = (
        "BAAI/bge-reranker-v2-m3"
    )

    DEFAULT_BATCH_SIZE = 8

    DEFAULT_MAX_LENGTH = 512

    def __init__(
        self,
        model_name: str = (
            DEFAULT_MODEL_NAME
        ),
        batch_size: int = (
            DEFAULT_BATCH_SIZE
        ),
        max_length: int = (
            DEFAULT_MAX_LENGTH
        ),
    ) -> None:
        self._model_name = (
            model_name
        )

        self.batch_size = (
            batch_size
        )

        self.max_length = (
            max_length
        )

        self.cuda_available = (
            torch.cuda.is_available()
        )

        self.device = (
            torch.device(
                "cuda:0"
            )
            if self.cuda_available
            else torch.device(
                "cpu"
            )
        )

        print(
            "[BgeReranker] "
            f"torch={torch.__version__}"
        )

        print(
            "[BgeReranker] "
            f"torch_cuda={torch.version.cuda}"
        )

        print(
            "[BgeReranker] "
            f"cuda_available="
            f"{self.cuda_available}"
        )

        print(
            "[BgeReranker] "
            f"device={self.device}"
        )

        if self.cuda_available:
            print(
                "[BgeReranker] "
                f"gpu="
                f"{torch.cuda.get_device_name(0)}"
            )

            torch.backends.cuda.matmul.allow_tf32 = (
                True
            )

            torch.set_float32_matmul_precision(
                "high"
            )

        else:
            print(
                "[BgeReranker][WARNING] "
                "CUDA를 사용할 수 없습니다. "
                "CPU로 실행됩니다."
            )

        print(
            "[BgeReranker] "
            f"model={self._model_name}"
        )

        print(
            "[BgeReranker] "
            f"batch_size={self.batch_size}"
        )

        print(
            "[BgeReranker] "
            f"max_length={self.max_length}"
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
    def name(
        self,
    ) -> str:
        return (
            "bge-reranker"
        )

    @property
    def model_name(
        self,
    ) -> str:
        return (
            self._model_name
        )

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

        total_started_at = (
            perf_counter()
        )

        scored_candidates: list[
            dict[str, Any]
        ] = []

        batch_count = (
            (
                len(candidates)
                + self.batch_size
                - 1
            )
            // self.batch_size
        )

        print(
            "[PERF][BgeReranker] "
            f"candidates={len(candidates)} "
            f"batch_size={self.batch_size} "
            f"batches={batch_count} "
            f"max_length={self.max_length} "
            f"device={self.device}"
        )

        for (
            batch_index,
            start,
        ) in enumerate(
            range(
                0,
                len(candidates),
                self.batch_size,
            ),
            start=1,
        ):
            batch_started_at = (
                perf_counter()
            )

            batch = candidates[
                start:
                start
                + self.batch_size
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

            tokenize_started_at = (
                perf_counter()
            )

            inputs = (
                self.tokenizer(
                    queries,
                    documents,
                    padding=True,
                    truncation=True,
                    max_length=(
                        self.max_length
                    ),
                    return_tensors="pt",
                )
            )

            tokenize_elapsed = (
                perf_counter()
                - tokenize_started_at
            )

            transfer_started_at = (
                perf_counter()
            )

            inputs = {
                key: value.to(
                    self.device,
                    non_blocking=(
                        self.cuda_available
                    ),
                )
                for (
                    key,
                    value,
                ) in inputs.items()
            }

            if self.cuda_available:
                torch.cuda.synchronize()

            transfer_elapsed = (
                perf_counter()
                - transfer_started_at
            )

            inference_started_at = (
                perf_counter()
            )

            with torch.inference_mode():
                if self.cuda_available:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        outputs = (
                            self.model(
                                **inputs
                            )
                        )
                else:
                    outputs = (
                        self.model(
                            **inputs
                        )
                    )

            if self.cuda_available:
                torch.cuda.synchronize()

            inference_elapsed = (
                perf_counter()
                - inference_started_at
            )

            logits = (
                outputs.logits
            )

            if (
                logits.ndim == 2
                and logits.shape[1] == 1
            ):
                scores = (
                    logits[:, 0]
                )

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
                .float()
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
                scored_candidates.append(
                    {
                        **candidate,
                        "rerank_score": (
                            float(
                                score
                            )
                        ),
                    }
                )

            batch_elapsed = (
                perf_counter()
                - batch_started_at
            )

            print(
                "[PERF][BgeReranker] "
                f"batch[{batch_index}/"
                f"{batch_count}] "
                f"size={len(batch)} "
                f"tokenize="
                f"{tokenize_elapsed * 1000:.0f}ms "
                f"transfer="
                f"{transfer_elapsed * 1000:.0f}ms "
                f"inference="
                f"{inference_elapsed * 1000:.0f}ms "
                f"total="
                f"{batch_elapsed * 1000:.0f}ms"
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

        total_elapsed = (
            perf_counter()
            - total_started_at
        )

        print(
            "[PERF][BgeReranker] "
            f"TOTAL="
            f"{total_elapsed * 1000:.0f}ms "
            f"candidates="
            f"{len(candidates)}"
        )

        return (
            scored_candidates[
                :top_k
            ]
        )