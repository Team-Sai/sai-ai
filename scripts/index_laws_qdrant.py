import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "law_chunks_bge_m3"

EMBEDDING_DIR = Path(
    "data/embeddings/bge-m3/laws"
)


def load_all_chunks() -> list[dict]:
    chunks = []

    embedding_files = sorted(
        EMBEDDING_DIR.glob("*.json")
    )

    if not embedding_files:
        raise FileNotFoundError(
            "BGE-M3 임베딩 파일이 없습니다."
        )

    for path in embedding_files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        chunks.extend(
            data.get("chunks", [])
        )

    return chunks


def main():
    client = QdrantClient(
        url=QDRANT_URL
    )

    chunks = load_all_chunks()

    if not chunks:
        print(
            "[종료] 적재할 Chunk가 없습니다."
        )
        return

    vector_size = len(
        chunks[0]["embedding"]
    )

    print(
        f"[Qdrant] collection="
        f"{COLLECTION_NAME}"
    )

    print(
        f"[Qdrant] vector_size="
        f"{vector_size}"
    )

    collection_exists = (
        client.collection_exists(
            collection_name=
                COLLECTION_NAME
        )
    )

    if collection_exists:
        print(
            "[Qdrant] 기존 collection 삭제"
        )

        client.delete_collection(
            collection_name=
                COLLECTION_NAME
        )

    client.create_collection(
        collection_name=
            COLLECTION_NAME,

        vectors_config=
            VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
    )

    print(
        "[Qdrant] collection 생성 완료"
    )

    points = []

    for index, chunk in enumerate(
        chunks
    ):
        metadata = (
            chunk.get(
                "metadata",
                {}
            )
        )

        payload = {
            "chunk_id":
                chunk["chunk_id"],

            "document_id":
                chunk["document_id"],

            "text":
                chunk["text"],

            "source_type":
                metadata.get(
                    "source_type"
                ),

            "document_kind":
                metadata.get(
                    "document_kind"
                ),

            "law_id":
                metadata.get(
                    "law_id"
                ),

            "law_name":
                metadata.get(
                    "law_name"
                ),

            "law_type":
                metadata.get(
                    "law_type"
                ),

            "ministry":
                metadata.get(
                    "ministry"
                ),

            "part":
                metadata.get(
                    "part"
                ),

            "chapter":
                metadata.get(
                    "chapter"
                ),

            "section":
                metadata.get(
                    "section"
                ),

            "subsection":
                metadata.get(
                    "subsection"
                ),

            "article_key":
                metadata.get(
                    "article_key"
                ),

            "article_number":
                metadata.get(
                    "article_number"
                ),

            "article_title":
                metadata.get(
                    "article_title"
                ),

            "effective_date":
                metadata.get(
                    "effective_date"
                ),

            "article_effective_date":
                metadata.get(
                    "article_effective_date"
                ),

            "promulgation_date":
                metadata.get(
                    "promulgation_date"
                ),

            "revision_type":
                metadata.get(
                    "revision_type"
                ),

            "source":
                metadata.get(
                    "source"
                ),
        }

        point = PointStruct(
            id=index,
            vector=
                chunk["embedding"],
            payload=payload,
        )

        points.append(
            point
        )

    batch_size = 100

    for start in range(
        0,
        len(points),
        batch_size,
    ):
        end = start + batch_size

        batch = points[
            start:end
        ]

        client.upsert(
            collection_name=
                COLLECTION_NAME,
            points=batch,
        )

        print(
            f"[Qdrant] "
            f"{min(end, len(points))}"
            f"/{len(points)} 적재"
        )

    print()
    print("=" * 50)
    print(
        f"Qdrant 적재 완료 "
        f"(전체={len(points)})"
    )


if __name__ == "__main__":
    main()