from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    모든 임베딩 모델이 따라야 하는 공통 인터페이스.

    RAG의 나머지 코드는
    실제 임베딩 모델이 Qwen인지 BGE인지 OpenAI인지
    알 필요가 없다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        pass