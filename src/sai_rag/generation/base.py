from abc import ABC, abstractmethod
from typing import Any


class ReviewGenerator(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_review(
        self,
        contract: dict[str, Any],
        rule_results: list[
            dict[str, Any]
        ],
        legal_reviews: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:
        pass