from abc import ABC, abstractmethod
from typing import Any


class RerankerProvider(ABC):
    """
    Retrieval 후보를 사용자 요청과의 실제 관련성을 기준으로
    다시 정렬하기 위한 공통 인터페이스.
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
    def rerank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        후보 문서를 재정렬하여
        관련도가 높은 문서부터 반환한다.
        """
        pass