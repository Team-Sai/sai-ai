from src.sai_rag.embedding.base import (
    EmbeddingProvider,
)
from src.sai_rag.embedding.qwen3 import (
    Qwen3Embedder,
)
from src.sai_rag.embedding.bge_m3 import (
    BgeM3Embedder,
)
from src.sai_rag.embedding.openai_embedder import (
    OpenAIEmbedder,
)


def create_embedder(
    provider: str,
) -> EmbeddingProvider:

    normalized_provider = (
        provider.strip().lower()
    )

    if normalized_provider == "qwen3":
        return Qwen3Embedder()

    if normalized_provider == "bge-m3":
        return BgeM3Embedder()

    if normalized_provider == "openai":
        return OpenAIEmbedder()

    raise ValueError(
        "지원하지 않는 embedding provider입니다: "
        f"{provider}"
    )