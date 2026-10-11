import torch
from sentence_transformers import SentenceTransformer

from src.sai_rag.embedding.base import EmbeddingProvider


class BgeM3Embedder(EmbeddingProvider):

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[BGE-M3] device={self.device}"
        )

        print(
            f"[BGE-M3] model={self.MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=self.device,
        )

    @property
    def name(self) -> str:
        return "bge-m3"

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()