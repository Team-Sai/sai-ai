import json
from pathlib import Path

from src.sai_rag.chunking.law_chunker import (
    LawChunker,
)


FILTERED_DIR = Path(
    "data/filtered/laws"
)

OUTPUT_DIR = Path(
    "data/chunks/laws"
)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_chunks(
    law_id: str,
    source: dict,
    chunks: list[dict],
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{law_id}.json"
    )

    output_data = {
        "schema_version": "1.0",
        "source": source,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def main():

    chunker = LawChunker()

    filtered_files = sorted(
        FILTERED_DIR.glob(
            "*.json"
        )
    )

    if not filtered_files:
        print(
            "[종료] 필터링된 법령 파일이 없습니다."
        )
        return

    success_count = 0
    total_chunk_count = 0

    for filtered_path in filtered_files:

        try:
            filtered_data = (
                load_json(
                    filtered_path
                )
            )

            source = (
                filtered_data[
                    "source"
                ]
            )

            documents = (
                filtered_data[
                    "documents"
                ]
            )

            chunks = (
                chunker.chunk_documents(
                    documents
                )
            )

            law_id = source["law_id"]
            law_name = source["law_name"]

            output_path = (
                save_chunks(
                    law_id=law_id,
                    source=source,
                    chunks=chunks,
                )
            )

            print()
            print(
                f"[성공] {law_name}"
            )
            print(
                f"  Document: "
                f"{len(documents)}"
            )
            print(
                f"  Chunk: "
                f"{len(chunks)}"
            )
            print(
                f"  저장: "
                f"{output_path}"
            )

            success_count += 1
            total_chunk_count += len(
                chunks
            )

        except Exception as e:

            print()
            print(
                f"[실패] "
                f"{filtered_path.name}"
            )
            print(
                f"  원인: {e}"
            )

    print()
    print("=" * 50)
    print(
        f"Chunking 완료 "
        f"(법령={success_count}, "
        f"전체 Chunk={total_chunk_count})"
    )


if __name__ == "__main__":
    main()