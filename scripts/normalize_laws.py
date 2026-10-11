import json
from collections import Counter
from pathlib import Path

from src.sai_rag.ingestion.normalizers.korean_law import KoreanLawNormalizer


RAW_DIR = Path(
    "data/raw/laws"
)

OUTPUT_DIR = Path(
    "data/normalized/laws"
)


def save_normalized(
    data: dict,
) -> Path:

    law_id = data["source"]["law_id"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{law_id}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def main():

    normalizer = (
        KoreanLawNormalizer()
    )

    raw_files = sorted(
        RAW_DIR.glob("*.json")
    )

    if not raw_files:
        print(
            "[종료] Raw 법령 파일이 없습니다."
        )
        return

    print(
        f"Raw 법령 {len(raw_files)}개 발견"
    )

    success_count = 0
    failure_count = 0

    for raw_path in raw_files:

        try:
            normalized = (
                normalizer.normalize_file(
                    raw_path
                )
            )

            output_path = (
                save_normalized(
                    normalized
                )
            )

            source = (
                normalized["source"]
            )

            documents = (
                normalized[
                    "documents"
                ]
            )

            counts = Counter(
                document[
                    "document_kind"
                ]
                for document
                in documents
            )

            print()
            print(
                f"[성공] "
                f"{source['law_name']}"
            )
            print(
                f"  ID: "
                f"{source['law_id']}"
            )
            print(
                f"  ARTICLE: "
                f"{counts.get('ARTICLE', 0)}"
            )
            print(
                f"  MAIN_TEXT: "
                f"{counts.get('MAIN_TEXT', 0)}"
            )
            print(
                f"  SUPPLEMENTARY: "
                f"{counts.get('SUPPLEMENTARY', 0)}"
            )
            print(
                f"  저장: {output_path}"
            )

            success_count += 1

        except Exception as e:

            print()
            print(
                f"[실패] "
                f"{raw_path.name}"
            )
            print(
                f"  원인: {e}"
            )

            failure_count += 1

    print()
    print("=" * 50)
    print(
        f"정규화 완료 "
        f"(성공={success_count}, "
        f"실패={failure_count})"
    )


if __name__ == "__main__":
    main()