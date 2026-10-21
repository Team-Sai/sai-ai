import argparse

from src.sai_rag.retrieval.law_retriever import (
    LawRetriever,
)


def print_results(
    question: str,
    results: list[dict],
) -> None:

    print()
    print("=" * 70)
    print(
        f"질문: {question}"
    )
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = (
            result["metadata"]
        )

        print()
        print(
            f"[{index}] "
            f"score="
            f"{result['score']:.4f}"
        )

        print(
            "  법령: "
            f"{metadata.get('law_name')}"
        )

        print(
            "  조문: "
            f"{metadata.get('article_number')}"
        )

        print(
            "  제목: "
            f"{metadata.get('article_title')}"
        )

        print(
            "  document_id: "
            f"{result['document_id']}"
        )

        print()
        print(
            result["text"]
        )

        print(
            "-" * 70
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--provider",
        default="bge-m3",
        choices=[
            "bge-m3",
            "qwen3",
            "openai",
        ],
    )

    args = parser.parse_args()

    retriever = LawRetriever(
        provider=args.provider
    )

    results = retriever.retrieve(
        question=args.question,
        top_k=args.top_k,
    )

    print_results(
        question=args.question,
        results=results,
    )


if __name__ == "__main__":
    main()