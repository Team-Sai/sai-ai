from typing import Any

from src.sai_rag.generation.base import (
    ReviewGenerator,
)
from src.sai_rag.retrieval.law_retriever import (
    LawRetriever,
)
from src.sai_rag.review.query_builder import (
    ContractReviewQueryBuilder,
)
from src.sai_rag.review.rule_engine import (
    ContractRuleEngine,
)


class ContractReviewService:

    def __init__(
        self,
        retriever: LawRetriever,
        generator: ReviewGenerator | None = None,
    ):
        self.retriever = retriever
        self.generator = generator

        self.rule_engine = (
            ContractRuleEngine()
        )

        self.query_builder = (
            ContractReviewQueryBuilder()
        )

    def review(
        self,
        contract: dict[str, Any],
        top_k: int = 3,
        generate: bool = True,
    ) -> dict[str, Any]:

        rule_results = (
            self.rule_engine.evaluate(
                contract
            )
        )

        queries = (
            self.query_builder.build_queries(
                contract
            )
        )

        rule_by_field = {
            result["field"]: result
            for result in rule_results
        }

        legal_reviews = []

        for query_info in queries:

            results = (
                self.retriever.retrieve(
                    question=
                        query_info["query"],
                    top_k=top_k,
                )
            )

            references = []

            for result in results:

                metadata = (
                    result["metadata"]
                )

                references.append(
                    {
                        "score":
                            result["score"],

                        "documentId":
                            result[
                                "document_id"
                            ],

                        "lawName":
                            metadata.get(
                                "law_name"
                            ),

                        "articleNumber":
                            metadata.get(
                                "article_number"
                            ),

                        "articleTitle":
                            metadata.get(
                                "article_title"
                            ),

                        "content":
                            result["text"],
                    }
                )

            field = query_info[
                "field"
            ]

            rule_result = (
                rule_by_field.get(
                    field
                )
            )

            legal_reviews.append(
                {
                    "field":
                        field,

                    "type":
                        query_info["type"],

                    "status":
                        (
                            rule_result[
                                "status"
                            ]
                            if rule_result
                            else "REVIEW"
                        ),

                    "ruleResult":
                        rule_result,

                    "query":
                        query_info["query"],

                    "references":
                        references,
                }
            )

        raw_result = {
            "ruleResults":
                rule_results,

            "legalReviews":
                legal_reviews,
        }

        if not generate:
            return raw_result

        if self.generator is None:
            raise RuntimeError(
                "generate=True인데 "
                "ReviewGenerator가 없습니다."
            )

        generated = (
            self.generator.generate_review(
                contract=contract,
                rule_results=
                    rule_results,
                legal_reviews=
                    legal_reviews,
            )
        )

        return {
            "review":
                generated,

            "debug":
                raw_result,
        }