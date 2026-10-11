from typing import Any

from sai_rag.agent.clause_generator import (
    ClauseGenerator,
)

from sai_rag.agent.clause_query_builder import (
    ClauseQueryBuilder,
)

from sai_rag.agent.clause_validator import (
    ClauseValidator,
)

from sai_rag.agent.issue_analyzer import (
    ClauseIssueAnalyzer,
)

from sai_rag.reranking.base import (
    RerankerProvider,
)

from sai_rag.reranking.hybrid_ranker import (
    HybridRanker,
)

from sai_rag.retrieval.law_retriever import (
    LawRetriever,
)


class ContractClauseAgent:
    """
    금전소비대차 계약 특약 작성 Agent.

    V4.3

    1. Issue Analysis
    2. Multi-query
    3. Metadata Filter
    4. Vector Retrieval
    5. Reranking
    6. Hybrid Ranking
    7. Clause Generation
    8. Domain Validation
    9. LLM Clause Validation
    10. User Intent Correction
    """

    def __init__(
        self,
        retriever: LawRetriever,
        generator: ClauseGenerator,
        reranker: RerankerProvider,
    ):
        self.retriever = (
            retriever
        )

        self.generator = (
            generator
        )

        self.reranker = (
            reranker
        )

        self.issue_analyzer = (
            ClauseIssueAnalyzer()
        )

        self.query_builder = (
            ClauseQueryBuilder()
        )

        self.validator = (
            ClauseValidator()
        )

        self.hybrid_ranker = (
            HybridRanker(
                rerank_weight=0.7,
                vector_weight=0.3,
            )
        )

    def suggest_clause(
        self,
        contract: dict[str, Any],
        message: str,
        candidate_top_k: int = 12,
        final_top_k: int = 5,
    ) -> dict[str, Any]:
        if not message.strip():
            raise ValueError(
                "특약 요청 내용을 입력해주세요."
            )

        # ======================================
        # 1. Issue Analysis
        # ======================================

        issues = (
            self.issue_analyzer.analyze(
                contract=contract,
                message=message,
            )
        )

        # ======================================
        # 2. Query Planning
        # ======================================

        search_plans = (
            self.query_builder.build_queries(
                contract=contract,
                message=message,
                issues=issues,
            )
        )

        collected_results: list[
            dict[str, Any]
        ] = []

        # ======================================
        # 3. Vector Retrieval
        # ======================================

        for plan in search_plans:
            query = str(
                plan.get(
                    "query",
                    "",
                )
            )

            if not query:
                continue

            law_names = (
                plan.get(
                    "lawNames"
                )
            )

            issue_type = (
                plan.get(
                    "issueType"
                )
            )

            results = (
                self.retriever.retrieve(
                    question=query,
                    top_k=(
                        candidate_top_k
                    ),
                    law_names=law_names,
                )
            )

            for result in results:
                collected_results.append(
                    {
                        **result,

                        "_issue_type": (
                            issue_type
                        ),

                        "_query": (
                            query
                        ),

                        "_rerank_query": (
                            f"{message.strip()} "
                            f"{query}"
                        ),
                    }
                )

        # ======================================
        # 4. 후보 병합
        # ======================================

        vector_candidates = (
            self._merge_vector_candidates(
                collected_results,
                max_results=20,
            )
        )

        # ======================================
        # 5. Reranking
        # ======================================

        reranked_candidates = (
            self.reranker.rerank(
                candidates=(
                    vector_candidates
                ),
                top_k=len(
                    vector_candidates
                ),
            )
        )

        # ======================================
        # 6. Hybrid Ranking
        # ======================================

        final_references = (
            self.hybrid_ranker.rank(
                candidates=(
                    reranked_candidates
                ),
                top_k=final_top_k,
            )
        )

        # ======================================
        # 7. Generation
        # ======================================

        generated = (
            self.generator.generate(
                contract=contract,
                message=message,
                issues=issues,
                references=(
                    final_references
                ),
            )
        )

        # ======================================
        # 8~9. Domain + LLM Validation
        # ======================================

        validation = (
            self.validator.validate(
                contract=contract,
                message=message,
                issues=issues,
                generated=generated,
                references=(
                    final_references
                ),
            )
        )

        final_clause = (
            validation.get(
                "revisedClause"
            )
            or generated.get(
                "suggestedClause"
            )
        )

        validator_warnings = (
            validation.get(
                "warnings",
                [],
            )
        )

        final_warnings = [
            str(warning).strip()
            for warning
            in validator_warnings
            if str(
                warning
            ).strip()
        ]

        if not final_warnings:
            final_warnings = [
                str(warning).strip()
                for warning
                in generated.get(
                    "warnings",
                    [],
                )
                if str(
                    warning
                ).strip()
            ]

        return {
            "message": (
                message
            ),

            "issues": [
                {
                    "type": (
                        issue.get(
                            "type"
                        )
                    ),
                    "reason": (
                        issue.get(
                            "reason"
                        )
                    ),
                }
                for issue
                in issues
            ],

            "title": (
                generated.get(
                    "title"
                )
            ),

            "summary": (
                generated.get(
                    "summary"
                )
            ),

            "suggestedClause": (
                final_clause
            ),

            "warnings": (
                final_warnings
            ),

            "validation": {
                "status": (
                    validation.get(
                        "status"
                    )
                ),

                "isSafeToSuggest": (
                    validation.get(
                        "isSafeToSuggest"
                    )
                ),

                "checks": (
                    validation.get(
                        "checks",
                        [],
                    )
                ),
            },

            "references": [
                self._to_public_reference(
                    reference
                )
                for reference
                in final_references
            ],

            "debug": {
                "pipelineVersion": (
                    "V4.3"
                ),

                "ranking": {
                    "strategy": (
                        "HYBRID"
                    ),

                    "rerankWeight": (
                        self.hybrid_ranker
                        .rerank_weight
                    ),

                    "vectorWeight": (
                        self.hybrid_ranker
                        .vector_weight
                    ),
                },

                "generatedClause": (
                    generated.get(
                        "suggestedClause"
                    )
                ),

                "validatedClause": (
                    validation.get(
                        "revisedClause"
                    )
                ),

                "requiresRevision": (
                    validation.get(
                        "requiresRevision"
                    )
                ),

                "unauthorizedAdditions": (
                    validation.get(
                        "unauthorizedAdditions",
                        [],
                    )
                ),

                "domainValidation": (
                    validation.get(
                        "domainValidation"
                    )
                ),

                "generatorWarnings": (
                    generated.get(
                        "warnings",
                        [],
                    )
                ),

                "validatorWarnings": (
                    validator_warnings
                ),

                "finalWarnings": (
                    final_warnings
                ),

                "searchPlans": [
                    {
                        "issueType": (
                            plan.get(
                                "issueType"
                            )
                        ),

                        "query": (
                            plan.get(
                                "query"
                            )
                        ),

                        "lawNames": (
                            plan.get(
                                "lawNames"
                            )
                        ),
                    }
                    for plan
                    in search_plans
                ],

                "candidateCount": (
                    len(
                        vector_candidates
                    )
                ),

                "vectorCandidates": [
                    self._to_vector_debug(
                        candidate
                    )
                    for candidate
                    in vector_candidates
                ],

                "rerankedCandidates": [
                    self._to_rerank_debug(
                        candidate
                    )
                    for candidate
                    in reranked_candidates
                ],

                "hybridCandidates": [
                    self._to_hybrid_debug(
                        candidate
                    )
                    for candidate
                    in final_references
                ],
            },
        }

    def _merge_vector_candidates(
        self,
        results: list[
            dict[str, Any]
        ],
        max_results: int = 20,
    ) -> list[
        dict[str, Any]
    ]:
        merged: dict[
            str,
            dict[str, Any],
        ] = {}

        for result in results:
            document_id = str(
                result.get(
                    "document_id",
                    "",
                )
            )

            if not document_id:
                continue

            existing = (
                merged.get(
                    document_id
                )
            )

            if existing is None:
                merged[
                    document_id
                ] = result
                continue

            if float(
                result.get(
                    "score",
                    0,
                )
            ) > float(
                existing.get(
                    "score",
                    0,
                )
            ):
                merged[
                    document_id
                ] = result

        return sorted(
            merged.values(),
            key=lambda item: float(
                item.get(
                    "score",
                    0,
                )
            ),
            reverse=True,
        )[:max_results]

    def _to_public_reference(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = reference.get(
            "metadata",
            {},
        )

        return {
            "documentId": (
                reference.get(
                    "document_id"
                )
            ),

            "lawName": (
                metadata.get(
                    "law_name"
                )
            ),

            "articleNumber": (
                metadata.get(
                    "article_number"
                )
            ),

            "articleTitle": (
                metadata.get(
                    "article_title"
                )
            ),

            "content": (
                reference.get(
                    "text",
                    ""
                )
            ),

            "vectorScore": (
                reference.get(
                    "score"
                )
            ),

            "rerankScore": (
                reference.get(
                    "rerank_score"
                )
            ),

            "finalScore": (
                reference.get(
                    "final_score"
                )
            ),

            "matchedIssue": (
                reference.get(
                    "_issue_type"
                )
            ),
        }

    def _to_vector_debug(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = reference.get(
            "metadata",
            {},
        )

        return {
            "documentId": (
                reference.get(
                    "document_id"
                )
            ),
            "lawName": (
                metadata.get(
                    "law_name"
                )
            ),
            "articleNumber": (
                metadata.get(
                    "article_number"
                )
            ),
            "articleTitle": (
                metadata.get(
                    "article_title"
                )
            ),
            "vectorScore": (
                reference.get(
                    "score"
                )
            ),
            "matchedIssue": (
                reference.get(
                    "_issue_type"
                )
            ),
        }

    def _to_rerank_debug(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        result = (
            self._to_vector_debug(
                reference
            )
        )

        result[
            "rerankScore"
        ] = reference.get(
            "rerank_score"
        )

        return result

    def _to_hybrid_debug(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        result = (
            self._to_rerank_debug(
                reference
            )
        )

        result[
            "vectorNormalized"
        ] = reference.get(
            "vector_normalized"
        )

        result[
            "rerankNormalized"
        ] = reference.get(
            "rerank_normalized"
        )

        result[
            "finalScore"
        ] = reference.get(
            "final_score"
        )

        return result