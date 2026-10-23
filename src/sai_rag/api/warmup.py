"""
서버가 켜질 때 검색·순위 모델을 미리 몇 번 돌려 두는 준비 운동(warm-up).

검색 임베딩(BGE-M3)과 순위 모델(BGE Reranker)은 처음 몇 번은
모델을 메모리에 올리고 계산 준비를 하느라 평소보다 몇 배 느리다.
(로컬 측정: 첫 검색 8.7초 → 두 번째 0.9초)

사용자의 첫 요청이 이 비용을 떠안지 않도록, 서버가 "준비 완료"가 되기 전에
가짜 질문으로 미리 돌려 둔다.

- OpenAI(LLM)는 부르지 않는다 → 추가 비용 없음. (우리 서버 CPU만 잠깐 씀)
- 실패해도 서버는 그대로 뜬다. (준비 운동일 뿐이라 첫 요청이 조금 느려질 뿐)
- AI_WARMUP_ROUNDS=0 이면 건너뛴다. (테스트·급하게 띄울 때)
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

WARMUP_QUERIES = (
    "만기 전에 원금을 먼저 갚을 수 있게 하고 싶어요",
    "연체하면 지연이자를 받고 싶어요",
)

DEFAULT_ROUNDS = 2


def warmup_rounds() -> int:
    value = os.getenv("AI_WARMUP_ROUNDS", "").strip()

    if not value:
        return DEFAULT_ROUNDS

    try:
        return max(0, int(value))
    except ValueError:
        print(
            "[Startup][WARNING] "
            f"AI_WARMUP_ROUNDS 값이 올바르지 않아 기본값({DEFAULT_ROUNDS})을 씁니다: {value!r}"
        )
        return DEFAULT_ROUNDS


def warm_up(retriever: Any, reranker: Any, rounds: int | None = None) -> int:
    """
    검색 → 순위 매기기를 rounds번 미리 돌린다. 성공한 횟수를 돌려준다.
    """

    rounds = warmup_rounds() if rounds is None else rounds
    succeeded = 0

    for index in range(rounds):
        query = WARMUP_QUERIES[index % len(WARMUP_QUERIES)]
        started_at = perf_counter()

        try:
            candidates = retriever.retrieve(question=query, top_k=5)

            if candidates and reranker is not None:
                reranker.rerank(candidates=candidates, top_k=len(candidates))

            succeeded += 1
            print(
                "[Startup] "
                f"warm-up {index + 1}/{rounds} "
                f"{(perf_counter() - started_at) * 1000:.0f}ms"
            )

        except Exception as exc:  # noqa: BLE001 - 준비 운동 실패로 서버가 안 뜨면 안 됨
            print(
                "[Startup][WARNING] "
                f"warm-up {index + 1}/{rounds} 실패: {exc!r}"
            )

    return succeeded
