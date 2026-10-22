from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Callable

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

ClauseEventCallback = Callable[[str, dict[str, Any]], None]

class ContractClauseAgent:
    """
    금전소비대차 계약 특약 작성 Agent.

    V5.4

    Pipeline

    1. Issue Analysis
    2. Query Planning
    3. Vector Retrieval
    4. Candidate Merge / Pre-pruning
    5. Reranking
    6. Hybrid Ranking
    7. Generation Context Pruning
    8. Clause Generation
    9. Deterministic / Conditional Validation

    Performance Strategy

    - Query별 Vector 후보: 8개
    - Reranker 입력: 최대 5개
    - 사용자에게 반환할 근거: 최대 5개
    - Generator가 실제 읽을 근거: 상위 3개
    """

    DEFAULT_CANDIDATE_TOP_K = 8

    MAX_RERANK_CANDIDATES = 5

    DEFAULT_FINAL_TOP_K = 5

    GENERATOR_REFERENCE_TOP_K = 3

    SLOT_PATTERN = re.compile(
        r"\[입력\s*필요\s*:\s*([^\[\]]+)\]"
    )

    def __init__(
        self,
        retriever: LawRetriever,
        generator: ClauseGenerator,
        reranker: RerankerProvider,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

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
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
        on_event: ClauseEventCallback | None = None,
    ) -> dict[str, Any]:
        """
        on_event를 넘기면 단계가 진행될 때마다 이벤트를 알려준다.
        (스트리밍 API에서 사용, 기존 호출은 넘기지 않으므로 동작이 같다)

        - ("progress", {"step": "analysis" | "retrieval" | "generation" | "validation"})
        - ("draft", 생성 직후 조항 초안 - 검증 전)
        """

        if not message.strip():
            raise ValueError(
                "특약 요청 내용을 입력해주세요."
            )

        resolved_candidate_top_k = (
            candidate_top_k
            if candidate_top_k is not None
            else self.DEFAULT_CANDIDATE_TOP_K
        )

        resolved_final_top_k = (
            final_top_k
            if final_top_k is not None
            else self.DEFAULT_FINAL_TOP_K
        )

        if resolved_candidate_top_k < 1:
            raise ValueError(
                "candidate_top_k는 1 이상이어야 합니다."
            )

        if resolved_final_top_k < 1:
            raise ValueError(
                "final_top_k는 1 이상이어야 합니다."
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

        print(
            "[PERF][ClauseAgent] "
            f"candidate_top_k="
            f"{resolved_candidate_top_k} "
            f"max_rerank_candidates="
            f"{self.MAX_RERANK_CANDIDATES} "
            f"final_top_k="
            f"{resolved_final_top_k} "
            f"generator_reference_top_k="
            f"{self.GENERATOR_REFERENCE_TOP_K}"
        )

        # ======================================
        # 1. Issue Analysis
        # ======================================

        self._emit(
            on_event,
            "progress",
            {"step": "analysis"},
        )

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
        # 1-1. Scope Guard
        # ======================================
        # 계약 특약과 관계없는 요청(예: 요리법)은
        # 법령 검색·리랭킹·생성을 하지 않고 바로 안내한다.
        if not issues:
            print(
                "[PERF][ClauseAgent] "
                "out_of_scope=true -> skip retrieval/rerank/generation"
            )

            return self._out_of_scope_response(
                message=message,
                issue_analysis_ms=(
                    performance["issue_analysis"] * 1000
                ),
            )

        # ======================================
        # 2. Query Planning
        # ======================================

        self._emit(
            on_event,
            "progress",
            {"step": "retrieval"},
        )

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
        # 3. Vector Retrieval
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

        for index, plan in enumerate(
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
                        resolved_candidate_top_k
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
                    "issueType": issue_type,
                    "query": query,
                    "lawNames": law_names,
                    "resultCount": len(
                        results
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
                max_results=(
                    self.MAX_RERANK_CANDIDATES
                ),
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

        print(
            "[PERF][ClauseAgent] "
            f"rerank_input_reduction="
            f"{len(collected_results)}"
            f"->{len(vector_candidates)}"
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
                        resolved_final_top_k
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
        # 7. Generator Context Pruning
        # ======================================

        generation_references = (
            final_references[
                :self.GENERATOR_REFERENCE_TOP_K
            ]
        )

        print(
            "[PERF][ClauseAgent] "
            f"generation_references="
            f"{len(generation_references)}"
            f"/{len(final_references)}"
        )

        # ======================================
        # 8. Clause Generation
        # ======================================

        self._emit(
            on_event,
            "progress",
            {"step": "generation"},
        )

        started_at = (
            perf_counter()
        )

        generated = (
            self.generator.generate(
                contract=contract,
                message=message,
                issues=issues,
                references=(
                    generation_references
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
        # 8-1. Draft 이벤트 (검증 전 조항 초안)
        # ======================================

        self._emit(
            on_event,
            "draft",
            {
                "title": generated.get("title"),
                "summary": generated.get("summary"),
                "suggestedClause": generated.get("suggestedClause"),
                "warnings": generated.get("warnings") or [],
                "unresolvedDecisions": (
                    generated.get("unresolvedDecisions") or []
                ),
                "references": [
                    self._to_public_reference(reference)
                    for reference in final_references
                ],
            },
        )

        self._emit(
            on_event,
            "progress",
            {"step": "validation"},
        )

        # ======================================
        # 9. Validation
        # ======================================
        #
        # Validator에는 최종 5개 전체를 전달한다.
        #
        # 평소에는 deterministic validation으로
        # 즉시 통과하지만,
        # LLM Validator가 필요한 예외 상황에서는
        # 더 넓은 근거를 확인할 수 있다.
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

        final_unresolved_decisions = (
            self._rebuild_unresolved_decisions(
                final_clause=final_clause,
                generated_decisions=(
                    generated.get(
                        "unresolvedDecisions",
                        [],
                    )
                ),
            )
        )

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

            # UI에는 최종 근거 5개 유지
            "references": [
                self._to_public_reference(
                    reference
                )
                for reference
                in final_references
            ],

            # Validator가 문장을 수정하면서 Slot을 추가/삭제할 수 있으므로
            # Generator의 목록을 그대로 사용하지 않고
            # 최종 조항에 실제 존재하는 Slot 기준으로 다시 구성한다.
            "unresolvedDecisions": (
                final_unresolved_decisions
            ),

            "debug": {
                "pipelineVersion": (
                    "V5.4"
                ),

                "validatorMode": (
                    validator_mode
                ),

                "performance": {
                    "issueAnalysisMs": round(
                        performance[
                            "issue_analysis"
                        ]
                        * 1000,
                        2,
                    ),

                    "queryPlanningMs": round(
                        performance[
                            "query_planning"
                        ]
                        * 1000,
                        2,
                    ),

                    "retrievalMs": round(
                        performance[
                            "retrieval"
                        ]
                        * 1000,
                        2,
                    ),

                    "candidateMergeMs": round(
                        performance[
                            "candidate_merge"
                        ]
                        * 1000,
                        2,
                    ),

                    "rerankingMs": round(
                        performance[
                            "reranking"
                        ]
                        * 1000,
                        2,
                    ),

                    "hybridRankingMs": round(
                        performance[
                            "hybrid_ranking"
                        ]
                        * 1000,
                        2,
                    ),

                    "generationMs": round(
                        performance[
                            "generation"
                        ]
                        * 1000,
                        2,
                    ),

                    "validationMs": round(
                        performance[
                            "validation"
                        ]
                        * 1000,
                        2,
                    ),

                    "totalMs": round(
                        performance[
                            "total"
                        ]
                        * 1000,
                        2,
                    ),

                    "retrievalDetails": (
                        retrieval_timings
                    ),
                },

                "candidateOptimization": {
                    "candidateTopK": (
                        resolved_candidate_top_k
                    ),

                    "rawCandidateCount": (
                        len(
                            collected_results
                        )
                    ),

                    "maxRerankCandidates": (
                        self.MAX_RERANK_CANDIDATES
                    ),

                    "rerankCandidateCount": (
                        len(
                            vector_candidates
                        )
                    ),

                    "finalReferenceCount": (
                        len(
                            final_references
                        )
                    ),

                    "generatorReferenceCount": (
                        len(
                            generation_references
                        )
                    ),
                },

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

    @staticmethod
    def _emit(
        on_event: ClauseEventCallback | None,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """
        이벤트를 알린다. 받는 쪽에서 오류가 나도 특약 생성은 계속한다.
        """

        if on_event is None:
            return

        try:
            on_event(event, data)
        except Exception as exc:  # noqa: BLE001 - 이벤트 전달 실패가 생성을 멈추면 안 됨
            print(
                "[ClauseAgent][WARNING] "
                f"event callback failed: {event} {exc!r}"
            )

    OUT_OF_SCOPE_GUIDE = (
        "사이원장 AI는 금전소비대차 계약서의 특약 작성만 도와드릴 수 있어요. "
        "이자, 상환 방법, 연체, 중도상환, 보증처럼 "
        "계약 조건에 대해 요청해 주세요."
    )

    def _out_of_scope_response(
        self,
        message: str,
        issue_analysis_ms: float,
    ) -> dict[str, Any]:
        """
        계약 특약과 무관한 요청에 대한 응답.

        기존 응답과 같은 필드를 유지해서 백엔드/프론트가
        그대로 받을 수 있게 한다.
        - suggestedClause는 비워 두고
        - validation.isSafeToSuggest=false + warnings로 사유를 전달한다.
          (백엔드 RagApiClient는 "조항 없음 + 거절 + 사유 있음"을 정상 응답으로 처리)
        """
        return {
            "message": message,
            "issues": [],
            "title": "요청 범위 밖이에요",
            "summary": self.OUT_OF_SCOPE_GUIDE,
            "suggestedClause": None,
            "warnings": [
                self.OUT_OF_SCOPE_GUIDE,
            ],
            "validation": {
                "status": "OUT_OF_SCOPE",
                "isSafeToSuggest": False,
                "checks": [
                    {
                        "type": "SCOPE",
                        "status": "REJECTED",
                        "message": (
                            "금전소비대차 계약 특약과 관련 없는 요청입니다."
                        ),
                    }
                ],
            },
            "references": [],
            "unresolvedDecisions": [],
            "debug": {
                "pipelineVersion": "V5.4",
                "outOfScope": True,
                "performance": {
                    "issueAnalysisMs": round(
                        issue_analysis_ms,
                        2,
                    ),
                },
            },
        }

    @classmethod
    def _extract_final_slots(
        cls,
        clause: str,
    ) -> list[str]:
        slots: list[str] = []

        for raw_name in (
            cls.SLOT_PATTERN.findall(
                clause
            )
        ):
            name = str(
                raw_name
            ).strip()

            if (
                name
                and name
                not in slots
            ):
                slots.append(
                    name
                )

        return slots

    @staticmethod
    def _normalize_slot_label(
        value: str,
    ) -> str:
        return re.sub(
            r"[\s·ㆍ,./()_\-]",
            "",
            value,
        ).lower()

    @classmethod
    def _rebuild_unresolved_decisions(
        cls,
        *,
        final_clause: str,
        generated_decisions: Any,
    ) -> list[dict[str, str]]:
        """
        Validator가 revisedClause를 반환할 수 있으므로
        unresolvedDecisions는 반드시 최종 조항의 Slot을 기준으로 만든다.

        Generator가 동일 Slot에 제공했던 reason이 있으면 재사용하고,
        Validator가 새 Slot을 추가했다면 일반 reason을 부여한다.
        """

        final_slots = (
            cls._extract_final_slots(
                final_clause
            )
        )

        generated_reason_by_name: dict[
            str,
            str,
        ] = {}

        if isinstance(
            generated_decisions,
            list,
        ):
            for item in (
                generated_decisions
            ):
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                name = str(
                    item.get(
                        "name",
                        "",
                    )
                ).strip()

                reason = str(
                    item.get(
                        "reason",
                        "",
                    )
                ).strip()

                if not name:
                    continue

                generated_reason_by_name[
                    cls._normalize_slot_label(
                        name
                    )
                ] = reason

        result: list[
            dict[str, str]
        ] = []

        for slot in final_slots:
            reason = (
                generated_reason_by_name.get(
                    cls._normalize_slot_label(
                        slot
                    )
                )
                or (
                    "최종 계약 조항에서 아직 확정되지 않은 "
                    "조건이므로 사용자가 값을 결정해야 합니다."
                )
            )

            result.append(
                {
                    "name": slot,
                    "reason": reason,
                }
            )

        return result

    def _merge_vector_candidates(
        self,
        results: list[
            dict[str, Any]
        ],
        max_results: int = 10,
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

        ranked_candidates = sorted(
            merged.values(),
            key=lambda item: float(
                item.get(
                    "score",
                    0,
                )
            ),
            reverse=True,
        )

        return ranked_candidates[
            :max_results
        ]

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
                and text
                not in result
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

        for group in (
            warning_groups
        ):
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