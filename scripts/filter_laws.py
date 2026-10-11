import json
from collections import Counter
from pathlib import Path

from src.sai_rag.ingestion.filters.law_domain import (
    LawDomainFilter,
)


NORMALIZED_DIR = Path(
    "data/normalized/laws"
)

OUTPUT_DIR = Path(
    "data/filtered/laws"
)

SCOPE_PATH = Path(
    "config/domain_scope.json"
)


def load_scope() -> dict:
    with SCOPE_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_filtered(
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

    scope_config = load_scope()

    domain_filter = (
        LawDomainFilter()
    )

    normalized_files = sorted(
        NORMALIZED_DIR.glob(
            "*.json"
        )
    )

    if not normalized_files:
        print(
            "[종료] 정규화된 법령이 없습니다."
        )
        return

    success_count = 0
    skip_count = 0

    for normalized_path in normalized_files:

        normalized_data = (
            load_json(
                normalized_path
            )
        )

        source = (
            normalized_data[
                "source"
            ]
        )

        law_id = source["law_id"]
        law_name = source["law_name"]

        scope = (
            scope_config.get(
                law_id
            )
        )

        if scope is None:
            print()
            print(
                f"[건너뜀] {law_name}"
            )
            print(
                f"  설정 없음: {law_id}"
            )

            skip_count += 1
            continue

        before_count = len(
            normalized_data[
                "documents"
            ]
        )

        filtered_data = (
            domain_filter
            .filter_documents(
                normalized_data=
                    normalized_data,
                scope=scope,
            )
        )

        documents = (
            filtered_data[
                "documents"
            ]
        )

        after_count = len(
            documents
        )

        counts = Counter(
            document[
                "document_kind"
            ]
            for document in documents
        )

        output_path = (
            save_filtered(
                filtered_data
            )
        )

        print()
        print(
            f"[성공] {law_name}"
        )
        print(
            f"  전체 문서: "
            f"{before_count}"
        )
        print(
            f"  필터링 후: "
            f"{after_count}"
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
            f"  저장: "
            f"{output_path}"
        )

        success_count += 1

    print()
    print("=" * 50)
    print(
        f"도메인 필터링 완료 "
        f"(성공={success_count}, "
        f"건너뜀={skip_count})"
    )


if __name__ == "__main__":
    main()