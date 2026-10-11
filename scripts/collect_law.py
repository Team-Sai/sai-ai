import json
from pathlib import Path

from src.sai_rag.ingestion.connectors.korean_law import KoreanLawConnector

OUTPUT_DIR = Path("data/raw/laws")

TARGET_LAWS = [
    "민법",
    "이자제한법",
    "이자제한법 제2조제1항의 최고이자율에 관한 규정",
    "민사집행법",
    "전자문서 및 전자거래 기본법",
    "전자서명법",
]


def save_json(file_name: str, data: dict):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = OUTPUT_DIR / file_name

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return file_path


def main():
    connector = KoreanLawConnector()

    for law_name in TARGET_LAWS:
        try:
            print(f"\n[검색] {law_name}")

            law = connector.find_exact_law(law_name)

            law_id = str(law["법령ID"])

            print(
                f"[본문 조회] {law_name} "
                f"(ID={law_id})"
            )

            document = connector.fetch_law(
                law_id
            )

            path = save_json(
                f"{law_id}.json",
                document,
            )

            print(f"[저장 완료] {path}")

        except Exception as e:
            print(
                f"[실패] {law_name}: {e}"
            )


if __name__ == "__main__":
    main()