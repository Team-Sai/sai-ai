import argparse
import json
from pathlib import Path

from src.sai_rag.embedding.factory import (
    create_embedder,
)


CHUNK_DIR = Path(
    "data/chunks/laws"
)

EMBEDDING_ROOT_DIR = Path(
    "data/embeddings"
)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_embeddings(
    output_dir: Path,
    source: dict,
    chunks: list[dict],
    vectors: list[list[float]],
    provider_name: str,
    model_name: str,
) -> Path:

    law_id = source["law_id"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedded_chunks = []

    for chunk, vector in zip(
        chunks,
        vectors,
    ):
        embedded_chunks.append(
            {
                "chunk_id":
                    chunk["chunk_id"],

                "document_id":
                    chunk["document_id"],

                "text":
                    chunk["text"],

                "metadata":
                    chunk["metadata"],

                "embedding":
                    vector,
            }
        )

    embedding_dimension = (
        len(vectors[0])
        if vectors
        else 0
    )

    output_data = {
        "schema_version": "1.0",

        "embedding_provider":
            provider_name,

        "embedding_model":
            model_name,

        "embedding_dimension":
            embedding_dimension,

        "source":
            source,

        "embedding_count":
            len(embedded_chunks),

        "chunks":
            embedded_chunks,
    }

    output_path = (
        output_dir
        / f"{law_id}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
        )

    return output_path


def main():

    parser = argparse.ArgumentParser(
        description=(
            "법률 Chunk를 "
            "Embedding Vector로 변환합니다."
        )
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "qwen3",
            "bge-m3",
            "openai",
        ],
        help=(
            "사용할 임베딩 provider"
        ),
    )

    args = parser.parse_args()

    embedder = create_embedder(
        args.provider
    )

    output_dir = (
        EMBEDDING_ROOT_DIR
        / embedder.name
        / "laws"
    )

    chunk_files = sorted(
        CHUNK_DIR.glob(
            "*.json"
        )
    )

    if not chunk_files:
        print(
            "[종료] Chunk 파일이 없습니다."
        )
        return

    print()
    print("=" * 50)
    print(
        f"Embedding Provider: "
        f"{embedder.name}"
    )
    print(
        f"Embedding Model: "
        f"{embedder.model_name}"
    )
    print("=" * 50)

    success_count = 0
    failure_count = 0
    total_count = 0

    for chunk_path in chunk_files:

        try:
            chunk_data = (
                load_json(
                    chunk_path
                )
            )

            source = (
                chunk_data[
                    "source"
                ]
            )

            chunks = (
                chunk_data[
                    "chunks"
                ]
            )

            law_id = (
                source["law_id"]
            )

            law_name = (
                source["law_name"]
            )

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            print()
            print(
                f"[임베딩 시작] "
                f"{law_name}"
            )

            vectors = (
                embedder.embed_documents(
                    texts
                )
            )

            if len(vectors) != len(chunks):
                raise ValueError(
                    "Chunk 수와 Embedding 수가 "
                    "일치하지 않습니다."
                )

            output_path = (
                save_embeddings(
                    output_dir=
                        output_dir,

                    source=
                        source,

                    chunks=
                        chunks,

                    vectors=
                        vectors,

                    provider_name=
                        embedder.name,

                    model_name=
                        embedder.model_name,
                )
            )

            dimension = (
                len(vectors[0])
                if vectors
                else 0
            )

            print(
                f"[성공] {law_name}"
            )
            print(
                f"  ID: {law_id}"
            )
            print(
                f"  Chunk: "
                f"{len(chunks)}"
            )
            print(
                f"  Embedding: "
                f"{len(vectors)}"
            )
            print(
                f"  Dimension: "
                f"{dimension}"
            )
            print(
                f"  저장: "
                f"{output_path}"
            )

            success_count += 1
            total_count += len(
                vectors
            )

        except Exception as e:

            print()
            print(
                f"[실패] "
                f"{chunk_path.name}"
            )

            print(
                f"  원인: {e}"
            )

            failure_count += 1

    print()
    print("=" * 50)

    print(
        "Embedding 완료 "
        f"(성공={success_count}, "
        f"실패={failure_count}, "
        f"전체={total_count})"
    )


if __name__ == "__main__":
    main()