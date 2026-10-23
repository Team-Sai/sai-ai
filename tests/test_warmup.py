"""
서버 시작 준비 운동(warm-up) 테스트.
"""

from sai_rag.api.warmup import warm_up, warmup_rounds


class FakeRetriever:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def retrieve(self, question, top_k=5):
        self.calls += 1
        if self.fail:
            raise RuntimeError("qdrant down")
        return [{"document_id": "law-1"}]


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, candidates, top_k=5):
        self.calls += 1
        return candidates


def test_warm_up_runs_retrieval_and_rerank_each_round():
    retriever, reranker = FakeRetriever(), FakeReranker()

    assert warm_up(retriever, reranker, rounds=2) == 2
    assert retriever.calls == 2
    assert reranker.calls == 2


def test_warm_up_failure_does_not_stop_server():
    retriever = FakeRetriever(fail=True)

    assert warm_up(retriever, FakeReranker(), rounds=2) == 0


def test_rounds_from_env(monkeypatch):
    monkeypatch.setenv("AI_WARMUP_ROUNDS", "0")
    assert warmup_rounds() == 0

    monkeypatch.setenv("AI_WARMUP_ROUNDS", "abc")
    assert warmup_rounds() == 2

    monkeypatch.delenv("AI_WARMUP_ROUNDS")
    assert warmup_rounds() == 2
