from __future__ import annotations

from time import perf_counter
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

    V5

    Pipeline

    1. Issue Analysis
    2. Query Planning
    3. Vector Retrieval
    4. Candidate Merge
    5. Reranking
    6. Hybrid Ranking
    7. Clause Generation
    8. Deterministic Validation
    9. Conditional LLM Validation
    """

    def __init__(
        self,
        retriever: LawRetriever,
        generator: ClauseGenerator,
        reranker: RerankerProvider,
    ) -> None:
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

        total_started_at = (
            perf_counter()
        )

        performance: dict[
            str,
            float,
        ] = {}

        print()
        print(
            "[PERF][ClauseAgent] "
            "========================================"
        )
        print(
            "[PERF][ClauseAgent] "
            "request started"
        )
        print(
            "[PERF][ClauseAgent] "
            f"message={message.strip()}"
        )

        # ======================================
        # 1. Issue Analysis
        # ======================================

        started_at = (
            perf_counter()
        )

        issues = (
            self.issue_analyzer.analyze(
                contract=contract,
                message=message,
            )
        )

        performance[
            "issue_analysis"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"issue_analysis="
            f"{performance['issue_analysis'] * 1000:.0f}ms"
        )

        # ======================================
        # 2. Query Planning
        # ======================================

        started_at = (
            perf_counter()
        )

        search_plans = (
            self.query_builder.build_queries(
                contract=contract,
                message=message,
                issues=issues,
            )
        )

        performance[
            "query_planning"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"query_planning="
            f"{performance['query_planning'] * 1000:.0f}ms "
            f"plans={len(search_plans)}"
        )

        # ======================================
        # 3. Retrieval
        # ======================================

        collected_results: list[
            dict[str, Any]
        ] = []

        retrieval_timings: list[
            dict[str, Any]
        ] = []

        retrieval_started_at = (
            perf_counter()
        )

        for (
            index,
            plan,
        ) in enumerate(
            search_plans,
            start=1,
        ):
            query = str(
                plan.get(
                    "query",
                    "",
                )
            ).strip()

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

            single_started_at = (
                perf_counter()
            )

            results = (
                self.retriever.retrieve(
                    question=query,
                    top_k=(
                        candidate_top_k
                    ),
                    law_names=(
                        law_names
                    ),
                )
            )

            elapsed = (
                perf_counter()
                - single_started_at
            )

            retrieval_timings.append(
                {
                    "index": index,
                    "issueType": (
                        issue_type
                    ),
                    "query": query,
                    "lawNames": (
                        law_names
                    ),
                    "resultCount": (
                        len(results)
                    ),
                    "elapsedMs": round(
                        elapsed * 1000,
                        2,
                    ),
                }
            )

            print(
                "[PERF][ClauseAgent] "
                f"retrieval[{index}]="
                f"{elapsed * 1000:.0f}ms "
                f"issue={issue_type} "
                f"results={len(results)}"
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

        performance[
            "retrieval"
        ] = (
            perf_counter()
            - retrieval_started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"retrieval_total="
            f"{performance['retrieval'] * 1000:.0f}ms "
            f"raw_candidates="
            f"{len(collected_results)}"
        )

        # ======================================
        # 4. Candidate Merge
        # ======================================

        started_at = (
            perf_counter()
        )

        vector_candidates = (
            self._merge_vector_candidates(
                collected_results,
                max_results=20,
            )
        )

        performance[
            "candidate_merge"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"candidate_merge="
            f"{performance['candidate_merge'] * 1000:.0f}ms "
            f"merged_candidates="
            f"{len(vector_candidates)}"
        )

        # ======================================
        # 5. Reranking
        # ======================================

        started_at = (
            perf_counter()
        )

        if vector_candidates:
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
        else:
            reranked_candidates = []

        performance[
            "reranking"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"reranking="
            f"{performance['reranking'] * 1000:.0f}ms "
            f"candidates="
            f"{len(vector_candidates)}"
        )

        # ======================================
        # 6. Hybrid Ranking
        # ======================================

        started_at = (
            perf_counter()
        )

        if reranked_candidates:
            final_references = (
                self.hybrid_ranker.rank(
                    candidates=(
                        reranked_candidates
                    ),
                    top_k=(
                        final_top_k
                    ),
                )
            )
        else:
            final_references = []

        performance[
            "hybrid_ranking"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"hybrid_ranking="
            f"{performance['hybrid_ranking'] * 1000:.0f}ms "
            f"final_references="
            f"{len(final_references)}"
        )

        # ======================================
        # 7. Clause Generation
        # ======================================

        started_at = (
            perf_counter()
        )

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

        performance[
            "generation"
        ] = (
            perf_counter()
            - started_at
        )

        print(
            "[PERF][ClauseAgent] "
            f"generation="
            f"{performance['generation'] * 1000:.0f}ms"
        )
        print(
            "[DEBUG][ClauseAgent] "
            f"generator_warnings="
            f"{generated.get('warnings', [])}"
        )

        print(
            "[DEBUG][ClauseAgent] "
            f"unresolved_decisions="
            f"{generated.get('unresolvedDecisions', [])}"
        )

        # ======================================
        # 8~9.
        # Deterministic Validation
        # +
        # Conditional LLM Validation
        # ======================================

        started_at = (
            perf_counter()
        )

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

        performance[
            "validation"
        ] = (
            perf_counter()
            - started_at
        )

        validator_mode = str(
            validation.get(
                "validatorMode",
                "UNKNOWN",
            )
        )

        print(
            "[PERF][ClauseAgent] "
            f"validation="
            f"{performance['validation'] * 1000:.0f}ms "
            f"mode={validator_mode}"
        )

        # ======================================
        # 10. Final Result
        # ======================================

        final_clause = str(
            validation.get(
                "revisedClause"
            )
            or validation.get(
                "suggestedClause"
            )
            or generated.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        validator_warnings = (
            self._normalize_warnings(
                validation.get(
                    "warnings"
                )
            )
        )

        generator_warnings = (
            self._normalize_warnings(
                generated.get(
                    "warnings"
                )
            )
        )

        final_warnings = (
            self._merge_warnings(
                validator_warnings,
                generator_warnings,
            )
        )

        validation_status = str(
            validation.get(
                "status",
                "REVIEW",
            )
        ).upper()

        validation_is_safe = bool(
            validation.get(
                "isSafeToSuggest",
                validation_status
                != "WARNING",
            )
        )

        validation_checks = (
            validation.get(
                "checks",
                [],
            )
        )

        if not isinstance(
            validation_checks,
            list,
        ):
            validation_checks = []

        performance[
            "total"
        ] = (
            perf_counter()
            - total_started_at
        )

        print(
            "[PERF][ClauseAgent] "
            "----------------------------------------"
        )
        print(
            "[PERF][ClauseAgent] "
            f"TOTAL="
            f"{performance['total'] * 1000:.0f}ms"
        )
        print(
            "[PERF][ClauseAgent] "
            "========================================"
        )
        print()

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
                    validation_status
                ),
                "isSafeToSuggest": (
                    validation_is_safe
                ),
                "checks": (
                    validation_checks
                ),
            },

            "references": [
                self._to_public_reference(
                    reference
                )
                for reference
                in final_references
            ],

            "unresolvedDecisions": (
                generated.get(
                    "unresolvedDecisions",
                    [],
                )
            ),

            "debug": {
                "pipelineVersion": (
                    "V5"
                ),

                "validatorMode": (
                    validator_mode
                ),

                "performance": {
                    "issueAnalysisMs": round(
                        performance[
                            "issue_analysis"
                        ] * 1000,
                        2,
                    ),
                    "queryPlanningMs": round(
                        performance[
                            "query_planning"
                        ] * 1000,
                        2,
                    ),
                    "retrievalMs": round(
                        performance[
                            "retrieval"
                        ] * 1000,
                        2,
                    ),
                    "candidateMergeMs": round(
                        performance[
                            "candidate_merge"
                        ] * 1000,
                        2,
                    ),
                    "rerankingMs": round(
                        performance[
                            "reranking"
                        ] * 1000,
                        2,
                    ),
                    "hybridRankingMs": round(
                        performance[
                            "hybrid_ranking"
                        ] * 1000,
                        2,
                    ),
                    "generationMs": round(
                        performance[
                            "generation"
                        ] * 1000,
                        2,
                    ),
                    "validationMs": round(
                        performance[
                            "validation"
                        ] * 1000,
                        2,
                    ),
                    "totalMs": round(
                        performance[
                            "total"
                        ] * 1000,
                        2,
                    ),
                    "retrievalDetails": (
                        retrieval_timings
                    ),
                },

                "ranking": {
                    "strategy": "HYBRID",
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
                    final_clause
                ),

                "requiresRevision": (
                    validation.get(
                        "requiresRevision",
                        False,
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

                "validatorDebug": (
                    validation.get(
                        "debug",
                        {},
                    )
                ),

                "generatorWarnings": (
                    generator_warnings
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

    # ==========================================
    # Candidate Merge
    # ==========================================

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

    # ==========================================
    # Warning
    # ==========================================

    @staticmethod
    def _normalize_warnings(
        value: Any,
    ) -> list[str]:
        if not isinstance(
            value,
            list,
        ):
            return []

        result: list[str] = []

        for item in value:
            text = str(
                item
            ).strip()

            if (
                text
                and text not in result
            ):
                result.append(
                    text
                )

        return result

    @staticmethod
    def _merge_warnings(
        *warning_groups: list[str],
    ) -> list[str]:
        result: list[str] = []

        for group in warning_groups:
            for warning in group:
                if (
                    warning
                    and warning
                    not in result
                ):
                    result.append(
                        warning
                    )

        return result

    # ==========================================
    # Public Reference
    # ==========================================

    def _to_public_reference(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = (
            reference.get(
                "metadata",
                {},
            )
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

    # ==========================================
    # Debug Reference
    # ==========================================

    def _to_vector_debug(
        self,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = (
            reference.get(
                "metadata",
                {},
            )
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