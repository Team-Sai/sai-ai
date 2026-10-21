from src.sai_rag.generation.base import (
    ReviewGenerator,
)
from src.sai_rag.generation.openai_generator import (
    OpenAIReviewGenerator,
)


def create_review_generator(
    provider: str,
) -> ReviewGenerator:

    normalized_provider = (
        provider.strip().lower()
    )

    if normalized_provider == "openai":
        return OpenAIReviewGenerator()

    raise ValueError(
        "지원하지 않는 "
        "Review Generator입니다: "
        f"{provider}"
    )