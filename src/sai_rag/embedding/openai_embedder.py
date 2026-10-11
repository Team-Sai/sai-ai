import os

from dotenv import load_dotenv
from openai import OpenAI

from src.sai_rag.embedding.base import EmbeddingProvider


load_dotenv()


class OpenAIEmbedder(EmbeddingProvider):

    MODEL_NAME = "text-embedding-3-small"

    def __init__(self):
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 없습니다. "
                ".env 파일을 확인해주세요."
            )

        self.client = OpenAI(
            api_key=api_key
        )

        print(
            f"[OpenAI] model={self.MODEL_NAME}"
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        response = (
            self.client.embeddings.create(
                model=self.MODEL_NAME,
                input=texts,
            )
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        response = (
            self.client.embeddings.create(
                model=self.MODEL_NAME,
                input=[text],
            )
        )

        return response.data[0].embedding