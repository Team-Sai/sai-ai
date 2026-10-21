import json

from src.sai_rag.generation.factory import (
    create_review_generator,
)
from src.sai_rag.retrieval.law_retriever import (
    LawRetriever,
)
from src.sai_rag.review.contract_review_service import (
    ContractReviewService,
)


def main():

    retriever = LawRetriever(
        provider="bge-m3"
    )

    generator = (
        create_review_generator(
            "openai"
        )
    )

    review_service = (
        ContractReviewService(
            retriever=retriever,
            generator=generator,
        )
    )

    contract = {
        "principal": 50_000_000,
        "interestRate": 25,
        "lateInterestRate": 30,
        "repaymentDate": "2028-09-30",
    }

    result = (
        review_service.review(
            contract=contract,
            top_k=3,
            generate=True,
        )
    )

    print()
    print("=" * 70)
    print(
        "AI 차용증 법률 검토 결과"
    )
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()