from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from sai_rag.agent.clause_generator import (
    ClauseGenerationCancelled,
)
from sai_rag.agent.openai_options import (
    openai_call_options,
)

from sai_rag.agent.intent_value_normalizer import (
    DATE_PATTERN,
    extra_value_tokens,
    find_amounts,
    find_dates,
    find_percents,
    is_value_reflected,
)

from sai_rag.agent.contract_domain_validator import (
    ContractDomainValidator,
)


load_dotenv()


# "2026년 12월 31일"에서 떨어져 나온 "2026년", "31일" 같은 날짜 조각
DATE_FRAGMENT_PATTERN = re.compile(r"\d+(?:년|월|일)")


class ClauseValidator:
    """
    생성된 계약 조항의 최종 안전 검증을 담당한다.

    V5.1

    핵심 전략:

    1. 확정적으로 판단 가능한 계약 데이터는
       ContractDomainValidator에서 Python 코드로 검증한다.

    2. 생성된 조항의 Slot / 모호 표현 / 구조적 문제를
       Python 코드로 먼저 검증한다.

    3. 정상적인 조항은 LLM Validator를 호출하지 않는다.

    4. 다음과 같은 경우에만 LLM Validator를 호출한다.
       - 미결정 조건이 Slot으로 표현되지 않음
       - 모호한 표현이 남음
       - 실제 위험성이 있는 Generator Warning
       - 직접적인 법률 판단/단정 표현이 포함됨
       - 일반 분류가 어려운 OTHER 쟁점
    """

    DEFAULT_MODEL = "gpt-6-luna"

    SLOT_PATTERN = re.compile(
        r"\[입력\s*필요\s*:\s*([^\[\]]+)\]"
    )

    AMBIGUOUS_PATTERNS: tuple[
        tuple[str, str],
        ...
    ] = (
        (
            "전부 또는 일부",
            "선택 범위가 확정되지 않았을 가능성이 있습니다.",
        ),
        (
            "일부 또는 전부",
            "선택 범위가 확정되지 않았을 가능성이 있습니다.",
        ),
        (
            "별도로 정한다",
            "실제 계약조건이 미정 상태로 남아 있습니다.",
        ),
        (
            "별도로 합의",
            "실제 계약조건이 미정 상태로 남아 있습니다.",
        ),
        (
            "추후 합의",
            "실제 계약조건이 미정 상태로 남아 있습니다.",
        ),
        (
            "당사자가 정한다",
            "계약 체결 시 확정해야 하는 조건을 뒤로 미루고 있습니다.",
        ),
        (
            "당사자가 합의",
            "계약 체결 시 확정해야 하는 조건을 뒤로 미루고 있습니다.",
        ),
        (
            "상당한 기간",
            "기간이 구체적으로 확정되지 않았습니다.",
        ),
        (
            "적절한 방법",
            "방법이 구체적으로 확정되지 않았습니다.",
        ),
        (
            "필요한 경우",
            "조건 발생 기준이 모호할 수 있습니다.",
        ),
    )

    LEGAL_ASSERTION_PATTERNS: tuple[
        str,
        ...
    ] = (
        "법적으로",
        "법률상",
        "법률에 따라",
        "무효이다",
        "무효로",
        "유효하다",
        "유효한",
        "위법",
        "적법",
        "강행규정",
        "법정이율",
    )

    RISKY_WARNING_PATTERNS: tuple[
        str,
        ...
    ] = (
        "법적 근거가 부족",
        "근거가 부족",
        "법령 근거가 부족",
        "검색된 법령만으로 판단하기 어렵",
        "검색 결과만으로 판단하기 어렵",
        "확인할 수 없",
        "법령에서 확인되지 않",
        "검색되지 않",
        "법적 판단이 어렵",
        "법률 판단이 어렵",
        "법적으로 불확실",
        "법률상 불확실",
        "법령과 충돌",
        "법률과 충돌",
        "계약정보와 충돌",
        "서로 충돌",
        "모순될 수",
        "법적 효력이 불확실",
        "유효성을 판단하기 어렵",
        "위법 가능",
        "무효 가능",
        "추가적인 법률 검토가 필요",
        "전문가 검토가 필요",
    )

    SUPPORTED_FAST_PATH_ISSUES = {
        "REPAYMENT",
        "DEFAULT",
        "ACCELERATION",
        "EARLY_REPAYMENT",
        "PENALTY",
        "LATE_INTEREST",
        "INTEREST",
        "GUARANTEE",
        "COLLATERAL",
    }

    # 단순한 "~했을 때", "~하면" 같은 일반 요청은
    # 그것만으로 LLM 검증을 수행하지 않는다.
    #
    # deterministic rule만으로 의미 보장이 어려운
    # 적용 범위 / 예외 / 순서 / 반복 조건만
    # semantic review 대상으로 둔다.
    SEMANTIC_INTENT_MARKERS = (
        "경우에만",
        "때에만",
        "에 한해서",
        "한해서",
        "오직",
        "제외",
        "제외하고",
        "다만",
        "연속",
        "원금만",
        "이자만",
        "일부만",
        "전액만",
        "이후에만",
        "이전에만",
        "보다 먼저",
        "보다 나중",
    )

    NEGATED_INTENT_KEYWORDS = (
        "수수료",
        "위약금",
        "연체이자",
        "지연손해금",
        "담보",
        "보증",
        "통지",
    )

    ISSUE_REQUIRED_DECISIONS: dict[
        str,
        list[str],
    ] = {
        "EARLY_REPAYMENT": [
            "중도상환 가능 범위",
            "최소 중도상환금액",
            "사전 통지 여부 및 기간",
            "통지 방법",
            "일부 상환 후 상환계획 조정 방식",
            "중도상환일까지의 이자 정산 방식",
            "중도상환 수수료 또는 손해배상 처리",
        ],
        "DEFAULT": [
            "연체 판단 기준",
            "일부 지급 시 연체 처리 기준",
        ],
        "ACCELERATION": [
            "일시변제 발생 요건",
            "연속 연체 판단 기준",
            "일부 지급 시 연체 처리 기준",
            "사전 통지 여부",
            "변제 유예기간",
        ],
        "PENALTY": [
            "위약금 산정 기준",
            "위약금 적용 대상 금액",
            "위약금 발생 횟수",
            "위약금 산정 기간",
            "위약금과 지연손해금의 관계",
        ],
        "LATE_INTEREST": [
            "연체이자율",
            "연체이자 계산 대상",
            "연체이자 계산 시작일",
        ],
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv("OPENAI_LLM_MODEL")
            or self.DEFAULT_MODEL
        )

        resolved_api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되어 있지 않습니다."
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
        )

        self.domain_validator = (
            ContractDomainValidator()
        )

        print(
            f"[ClauseValidator] model={self.model}"
        )

    def validate(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[dict[str, Any]],
        generated: dict[str, Any],
        references: list[dict[str, Any]],
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Generator 결과를 검증한다.

        should_stop()이 True면(사용자가 취소) LLM 검증을 부르기 직전에 멈춘다.

        정상적인 경우:
            deterministic validation만 수행한다.

        구조적 또는 법적 추가 판단이 필요한 경우:
            LLM Validator를 수행한다.
        """

        original_clause = str(
            generated.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        generator_warnings = (
            self._normalize_warnings(
                generated.get(
                    "warnings"
                )
            )
        )

        unresolved_decisions = (
            self._normalize_unresolved_decisions(
                generated.get(
                    "unresolvedDecisions"
                )
            )
        )

        issue_types = [
            str(
                issue.get(
                    "type",
                    "OTHER",
                )
            ).upper()
            for issue in issues
        ]

        # ======================================
        # 1. 계약 기본 데이터 검증
        # ======================================

        domain_validation = (
            self.domain_validator.validate(
                contract
            )
        )

        # ======================================
        # 2. 생성 조항 구조 검증
        # ======================================

        existing_slots = (
            self._extract_slots(
                original_clause
            )
        )

        ambiguous_matches = (
            self._find_unresolved_ambiguous_phrases(
                clause=original_clause,
                message=message,
            )
        )

        missing_slots = (
            self._find_missing_slots(
                unresolved_decisions=(
                    unresolved_decisions
                ),
                existing_slots=existing_slots,
            )
        )

        legal_assertions = (
            self._find_legal_assertions(
                original_clause
            )
        )

        risky_generator_warnings = (
            self._find_risky_generator_warnings(
                generator_warnings
            )
        )

        # ======================================
        # 생성 조항 ↔ 계약 확정값 / 사용자 요청
        # 실제 내용 일치 여부 검증
        # ======================================

        contract_clause_conflicts = (
            self._find_contract_clause_conflicts(
                contract=contract,
                clause=original_clause,
            )
        )

        user_intent_conflicts = (
            self._find_user_intent_conflicts(
                message=message,
                clause=original_clause,
                contract=contract,
            )
        )

        semantic_intent_review = (
            self._requires_semantic_intent_review(
                message=message,
                conflicts=(
                    user_intent_conflicts
                ),
            )
        )

        print(
            "[ClauseValidator] "
            f"contract_clause_conflicts="
            f"{contract_clause_conflicts}"
        )

        print(
            "[ClauseValidator] "
            f"user_intent_conflicts="
            f"{user_intent_conflicts}"
        )

        print(
            "[ClauseValidator] "
            f"semantic_intent_review="
            f"{semantic_intent_review}"
        )

        print(
            "[ClauseValidator] "
            f"generator_warnings="
            f"{generator_warnings}"
        )

        print(
            "[ClauseValidator] "
            f"risky_generator_warnings="
            f"{risky_generator_warnings}"
        )

        print(
            "[ClauseValidator] "
            f"existing_slots="
            f"{existing_slots}"
        )

        print(
            "[ClauseValidator] "
            f"missing_slots="
            f"{missing_slots}"
        )

        print(
            "[ClauseValidator] "
            f"ambiguous_matches="
            f"{ambiguous_matches}"
        )

        # ======================================
        # 3. 계약 기본정보에 명백한 오류
        # ======================================

        if (
            domain_validation.get(
                "status"
            )
            == "WARNING"
        ):
            print(
                "[ClauseValidator] "
                "mode=DETERMINISTIC_BLOCK "
                "reason=contract_domain_warning"
            )

            warnings = list(
                generator_warnings
            )

            for check in (
                domain_validation.get(
                    "fieldChecks",
                    [],
                )
            ):
                if (
                    check.get(
                        "status"
                    )
                    != "WARNING"
                ):
                    continue

                warning_text = str(
                    check.get(
                        "message",
                        "",
                    )
                ).strip()

                if (
                    warning_text
                    and warning_text
                    not in warnings
                ):
                    warnings.append(
                        warning_text
                    )

            return self._build_result(
                suggested_clause=(
                    original_clause
                ),
                original_clause=(
                    original_clause
                ),
                warnings=warnings,
                status="WARNING",
                is_safe=False,
                checks=[
                    self._domain_consistency_check(
                        domain_validation
                    ),
                    {
                        "type": "GROUNDING",
                        "status": "REVIEW",
                        "message": (
                            "계약 기본정보 오류가 있어 "
                            "법령 근거 검증 이전에 "
                            "계약정보 확인이 필요합니다."
                        ),
                    },
                    {
                        "type": "USER_INTENT",
                        "status": "REVIEW",
                        "message": (
                            "계약 기본정보를 수정한 뒤 "
                            "사용자 요청 반영 여부를 "
                            "다시 확인해야 합니다."
                        ),
                    },
                    {
                        "type": "AMBIGUITY",
                        "status": (
                            "REVIEW"
                            if (
                                ambiguous_matches
                                or missing_slots
                            )
                            else "PASS"
                        ),
                        "message": (
                            "미정 계약조건 확인이 필요합니다."
                            if (
                                ambiguous_matches
                                or missing_slots
                            )
                            else
                            "구조적으로 명확한 형태의 "
                            "조항입니다."
                        ),
                    },
                ],
                domain_validation=(
                    domain_validation
                ),
                validator_mode=(
                    "DETERMINISTIC_BLOCK"
                ),
                ambiguous_matches=(
                    ambiguous_matches
                ),
                slots=existing_slots,
                missing_slots=missing_slots,
                legal_assertions=(
                    legal_assertions
                ),
                llm_reasons=[
                    "contract_domain_warning"
                ],
            )

        # ======================================
        # 4. 검색 근거가 아예 없는 경우
        # ======================================

        if not references:
            print(
                "[ClauseValidator] "
                "mode=DETERMINISTIC_BLOCK "
                "reason=no_references"
            )

            warnings = list(
                generator_warnings
            )

            warning = (
                "조항을 뒷받침할 법령 근거를 "
                "검색하지 못해 추가 확인이 필요합니다."
            )

            if warning not in warnings:
                warnings.append(
                    warning
                )

            return self._build_result(
                suggested_clause=(
                    original_clause
                ),
                original_clause=(
                    original_clause
                ),
                warnings=warnings,
                status="WARNING",
                is_safe=False,
                checks=[
                    {
                        "type": "GROUNDING",
                        "status": "WARNING",
                        "message": (
                            "검색된 법령 근거가 없습니다."
                        ),
                    },
                    {
                        "type": "USER_INTENT",
                        "status": "REVIEW",
                        "message": (
                            "사용자 요청은 반영되었을 수 있으나 "
                            "법적 근거 확인이 필요합니다."
                        ),
                    },
                    self._domain_consistency_check(
                        domain_validation
                    ),
                    {
                        "type": "AMBIGUITY",
                        "status": (
                            "REVIEW"
                            if (
                                ambiguous_matches
                                or missing_slots
                            )
                            else "PASS"
                        ),
                        "message": (
                            "미정 계약조건 확인이 필요합니다."
                            if (
                                ambiguous_matches
                                or missing_slots
                            )
                            else
                            "구조적으로 명확한 형태의 "
                            "조항입니다."
                        ),
                    },
                ],
                domain_validation=(
                    domain_validation
                ),
                validator_mode=(
                    "DETERMINISTIC_BLOCK"
                ),
                ambiguous_matches=(
                    ambiguous_matches
                ),
                slots=existing_slots,
                missing_slots=missing_slots,
                legal_assertions=(
                    legal_assertions
                ),
                llm_reasons=[
                    "no_references"
                ],
            )

        # ======================================
        # 5. LLM Validator 필요 여부
        # ======================================

        llm_reasons = (
            self._collect_llm_reasons(
                issue_types=issue_types,
                risky_generator_warnings=(
                    risky_generator_warnings
                ),
                ambiguous_matches=(
                    ambiguous_matches
                ),
                missing_slots=missing_slots,
                legal_assertions=(
                    legal_assertions
                ),
                original_clause=(
                    original_clause
                ),
                contract_clause_conflicts=(
                    contract_clause_conflicts
                ),
                user_intent_conflicts=(
                    user_intent_conflicts
                ),
                semantic_intent_review=(
                    semantic_intent_review
                ),
            )
        )

        # ======================================
        # 6. Fast Path
        # ======================================

        if not llm_reasons:
            print(
                "[ClauseValidator] "
                "mode=DETERMINISTIC "
                "llm_skipped=true"
            )

            checks = [
                {
                    "type": "GROUNDING",
                    "status": "PASS",
                    "message": (
                        "검색 법령 근거가 존재하며 "
                        "추가적인 법률 재검증이 필요한 "
                        "위험 신호가 감지되지 않았습니다."
                    ),
                },
                {
                    "type": "USER_INTENT",
                    "status": "PASS",
                    "message": (
                        "사용자가 명시한 수치·날짜·부정 조건에서 "
                        "생성 조항과의 충돌이 발견되지 않았고, "
                        "추가적인 의미 검증이 필요한 자유 조건도 없습니다."
                    ),
                },
                {
                    "type": "CONTRACT_CONSISTENCY",
                    "status": "PASS",
                    "message": (
                        "생성 조항에 명시된 원금·만기일·약정이율·"
                        "정기 상환일·상환방식이 현재 계약의 "
                        "확정값과 충돌하지 않습니다."
                    ),
                },
                (
                    {
                        "type": "AMBIGUITY",
                        "status": "WARNING",
                        "message": (
                            "조항에 모호한 표현이 있어요: "
                            + ", ".join(
                                f"'{match['phrase']}'"
                                for match in ambiguous_matches
                            )
                            + ". 구체적인 기간·조건으로 바꾸거나 "
                            "[입력 필요: …] 칸으로 남겨 주세요."
                        ),
                    }
                    if ambiguous_matches
                    else {
                        "type": "AMBIGUITY",
                        "status": "PASS",
                        "message": (
                            "미정 계약조건은 입력 Slot으로 "
                            "구조화되어 있으며 "
                            "추가적인 모호 표현이 "
                            "감지되지 않았습니다."
                        ),
                    }
                ),
            ]

            fast_path_status = (
                "WARNING"
                if ambiguous_matches
                else "PASS"
            )

            fast_path_warnings = list(
                generator_warnings
            ) + [
                f"'{match['phrase']}' 표현: {match['reason']} "
                "구체적으로 정해 주세요."
                for match in ambiguous_matches
            ]

            return self._build_result(
                suggested_clause=(
                    original_clause
                ),
                original_clause=(
                    original_clause
                ),
                warnings=(
                    fast_path_warnings
                ),
                status=fast_path_status,
                is_safe=True,
                checks=checks,
                domain_validation=(
                    domain_validation
                ),
                validator_mode=(
                    "DETERMINISTIC"
                ),
                ambiguous_matches=(
                    ambiguous_matches
                ),
                slots=existing_slots,
                missing_slots=missing_slots,
                legal_assertions=(
                    legal_assertions
                ),
                llm_reasons=[],
            )

        # ======================================
        # 7. 문제가 있을 때만 LLM Validator
        # ======================================

        print(
            "[ClauseValidator] "
            "mode=LLM "
            f"reasons={','.join(llm_reasons)}"
        )

        decision_candidates = (
            self._collect_decision_candidates(
                issue_types
            )
        )

        system_prompt = (
            self._build_system_prompt()
        )

        user_prompt = (
            self._build_user_prompt(
                message=message,
                contract=contract,
                issues=issues,
                generated=generated,
                references=references,
                ambiguous_matches=(
                    ambiguous_matches
                ),
                decision_candidates=(
                    decision_candidates
                ),
                existing_slots=(
                    existing_slots
                ),
                missing_slots=(
                    missing_slots
                ),
                domain_validation=(
                    domain_validation
                ),
                llm_reasons=(
                    llm_reasons
                ),
                contract_clause_conflicts=(
                    contract_clause_conflicts
                ),
                user_intent_conflicts=(
                    user_intent_conflicts
                ),
                semantic_intent_review=(
                    semantic_intent_review
                ),
            )
        )

        # 검증 LLM은 약 12초가 걸리므로, 부르기 직전에 한 번 더 취소 여부를 본다.
        if should_stop is not None and should_stop():
            raise ClauseGenerationCancelled()

        response = (
            self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                **openai_call_options("VALIDATOR")
            )
        )

        raw_text = (
            self._extract_text(
                response
            )
        )

        validated = (
            self._parse_json(
                raw_text
            )
        )

        return self._normalize_llm_result(
            validated=validated,
            generated=generated,
            original_clause=(
                original_clause
            ),
            domain_validation=(
                domain_validation
            ),
            ambiguous_matches=(
                ambiguous_matches
            ),
            existing_slots=(
                existing_slots
            ),
            missing_slots=(
                missing_slots
            ),
            legal_assertions=(
                legal_assertions
            ),
            llm_reasons=(
                llm_reasons
            ),
        )

    # ==========================================
    # LLM 실행 여부 판단
    # ==========================================

    def _collect_llm_reasons(
        self,
        *,
        issue_types: list[str],
        risky_generator_warnings: list[str],
        ambiguous_matches: list[
            dict[str, str]
        ],
        missing_slots: list[str],
        legal_assertions: list[str],
        original_clause: str,
        contract_clause_conflicts: list[str],
        user_intent_conflicts: list[str],
        semantic_intent_review: bool,
    ) -> list[str]:
        reasons: list[str] = []

        if not original_clause:
            reasons.append(
                "empty_clause"
            )

        if any(
            issue_type
            not in self.SUPPORTED_FAST_PATH_ISSUES
            for issue_type in issue_types
        ):
            reasons.append(
                "unsupported_or_other_issue"
            )

        if risky_generator_warnings:
            reasons.append(
                "risky_generator_warning"
            )

        # 모호 표현(ambiguous_matches)은 규칙으로 정확히 찾을 수 있으므로
        # LLM 재검증 사유로 쓰지 않는다. (Fast Path에서 WARNING으로 안내)

        if missing_slots:
            reasons.append(
                "missing_required_slot"
            )

        if legal_assertions:
            reasons.append(
                "legal_assertion"
            )

        if contract_clause_conflicts:
            reasons.append(
                "contract_value_conflict"
            )

        if user_intent_conflicts:
            reasons.append(
                "user_intent_conflict"
            )

        if semantic_intent_review:
            reasons.append(
                "free_text_intent_requires_semantic_review"
            )

        return reasons

    def _find_risky_generator_warnings(
        self,
        warnings: list[str],
    ) -> list[str]:
        """
        Generator warning 전체를 LLM 재검증 사유로 보지 않는다.

        다음과 같은 warning은 그대로 사용자에게 보여주되
        LLM Validator를 다시 호출하지 않는다.

        - 사용자가 결정해야 하는 계약조건 안내
        - Slot 입력 필요 안내
        - 일반적인 법률상 주의사항

        실제 근거 부족, 법적 충돌, 효력 불확실성 등의
        위험 신호만 LLM Validator 재검증 대상으로 삼는다.
        """

        if not warnings:
            return []

        risky_warnings: list[str] = []

        for warning in warnings:
            normalized = str(
                warning
            ).strip()

            if not normalized:
                continue

            if any(
                pattern in normalized
                for pattern
                in self.RISKY_WARNING_PATTERNS
            ):
                risky_warnings.append(
                    normalized
                )

        return risky_warnings

    # ==========================================
    # Deterministic 검사
    # ==========================================

    @staticmethod
    def _first_contract_value(
        contract: dict[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            value = contract.get(
                key
            )

            if value is None:
                continue

            if isinstance(
                value,
                str,
            ):
                value = value.strip()

                if not value:
                    continue

            return value

        return None

    @staticmethod
    def _normalize_money(
        value: Any,
    ) -> int | None:
        try:
            return int(
                round(
                    float(value)
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _normalize_rate(
        value: Any,
    ) -> float | None:
        try:
            return round(
                float(value),
                6,
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _normalize_date_text(
        value: str,
    ) -> str | None:
        match = re.search(
            r"(20\d{2})\s*[-./년]\s*"
            r"(\d{1,2})\s*[-./월]\s*"
            r"(\d{1,2})\s*일?",
            str(value),
        )

        if not match:
            return None

        year = int(
            match.group(1)
        )
        month = int(
            match.group(2)
        )
        day = int(
            match.group(3)
        )

        return (
            f"{year:04d}-"
            f"{month:02d}-"
            f"{day:02d}"
        )

    def _find_contract_clause_conflicts(
        self,
        *,
        contract: dict[str, Any],
        clause: str,
    ) -> list[str]:
        """
        생성 조항 안에서 '현재 계약의 확정값'이라고 해석할 수 있는
        값만 골라 실제 계약정보와 비교한다.

        위약금 2,000원처럼 새 조항 자체의 금액은 원금과 비교하지 않고,
        '원금/대여금은 O원', '최종 만기일은 O'처럼 기존 계약 필드를
        직접 다시 선언하는 표현만 검사한다.
        """

        conflicts: list[str] = []

        principal = (
            self._normalize_money(
                self._first_contract_value(
                    contract,
                    "principalAmount",
                    "principal",
                )
            )
        )

        if principal is not None:
            principal_pattern = re.compile(
                r"(?:대여\s*원금|대여금|차용금|원금)"
                r"[^0-9]{0,18}"
                r"([0-9][0-9,]*)\s*원"
            )

            for match in (
                principal_pattern.finditer(
                    clause
                )
            ):
                clause_principal = (
                    self._normalize_money(
                        match.group(1).replace(
                            ",",
                            "",
                        )
                    )
                )

                if (
                    clause_principal
                    is not None
                    and clause_principal
                    != principal
                ):
                    conflicts.append(
                        (
                            "생성 조항이 현재 대여원금 "
                            f"{principal}원과 다른 "
                            f"{clause_principal}원을 "
                            "원금/대여금으로 명시했습니다."
                        )
                    )

        maturity_date = (
            self._first_contract_value(
                contract,
                "maturityDate",
                "repaymentDate",
            )
        )

        normalized_maturity = (
            self._normalize_date_text(
                str(maturity_date)
            )
            if maturity_date
            else None
        )

        if normalized_maturity:
            maturity_pattern = re.compile(
                r"(?:최종\s*)?"
                r"(?:만기일|변제기일|변제기)"
                r"[^0-9]{0,18}"
                r"(20\d{2}\s*[-./년]\s*"
                r"\d{1,2}\s*[-./월]\s*"
                r"\d{1,2}\s*일?)"
            )

            for match in (
                maturity_pattern.finditer(
                    clause
                )
            ):
                clause_date = (
                    self._normalize_date_text(
                        match.group(1)
                    )
                )

                if (
                    clause_date
                    and clause_date
                    != normalized_maturity
                ):
                    conflicts.append(
                        (
                            "생성 조항이 현재 최종 만기일 "
                            f"{normalized_maturity}과 다른 "
                            f"{clause_date}을 만기/변제기로 "
                            "명시했습니다."
                        )
                    )

        interest_rate = (
            self._normalize_rate(
                self._first_contract_value(
                    contract,
                    "interestRate",
                )
            )
        )

        if interest_rate is not None:
            interest_pattern = re.compile(
                r"(?<!연체)"
                r"(?:약정\s*이율|약정\s*이자율|이자율)"
                r"[^0-9]{0,12}"
                r"([0-9]+(?:\.[0-9]+)?)\s*%"
            )

            for match in (
                interest_pattern.finditer(
                    clause
                )
            ):
                clause_rate = (
                    self._normalize_rate(
                        match.group(1)
                    )
                )

                if (
                    clause_rate
                    is not None
                    and clause_rate
                    != interest_rate
                ):
                    conflicts.append(
                        (
                            "생성 조항이 현재 약정이자율 "
                            f"{interest_rate}%와 다른 "
                            f"{clause_rate}%를 약정이율로 "
                            "명시했습니다."
                        )
                    )

        repayment_day = (
            self._first_contract_value(
                contract,
                "repaymentDay",
            )
        )

        try:
            normalized_repayment_day = (
                int(repayment_day)
                if repayment_day is not None
                else None
            )
        except (
            TypeError,
            ValueError,
        ):
            normalized_repayment_day = None

        if normalized_repayment_day is not None:
            repayment_day_pattern = re.compile(
                r"(?:정기\s*)?"
                r"(?:상환일|변제일)"
                r"[^0-9]{0,12}"
                r"(?:매월\s*)?"
                r"(\d{1,2})\s*일"
            )

            for match in (
                repayment_day_pattern.finditer(
                    clause
                )
            ):
                clause_day = int(
                    match.group(1)
                )

                if (
                    clause_day
                    != normalized_repayment_day
                ):
                    conflicts.append(
                        (
                            "생성 조항이 현재 정기 상환일 "
                            f"매월 {normalized_repayment_day}일과 "
                            f"다른 매월 {clause_day}일을 "
                            "상환일로 명시했습니다."
                        )
                    )

        repayment_type = str(
            self._first_contract_value(
                contract,
                "repaymentType",
            )
            or ""
        ).strip()

        repayment_labels = {
            "EQUAL_PRINCIPAL_AND_INTEREST": (
                "원리금균등상환"
            ),
            "EQUAL_PRINCIPAL": (
                "원금균등상환"
            ),
            "BULLET": (
                "만기일시상환"
            ),
            "LUMP_SUM": (
                "일시상환"
            ),
        }

        if repayment_type:
            mentioned_types = [
                (
                    code,
                    label,
                )
                for (
                    code,
                    label,
                )
                in repayment_labels.items()
                if label in clause
            ]

            for (
                code,
                label,
            ) in mentioned_types:
                if (
                    code
                    != repayment_type
                ):
                    conflicts.append(
                        (
                            "생성 조항이 현재 상환방식 "
                            f"{repayment_labels.get(repayment_type, repayment_type)}과 "
                            f"다른 {label} 방식을 명시했습니다."
                        )
                    )

        return (
            self._deduplicate_strings(
                conflicts
            )
        )

    @staticmethod
    def _extract_explicit_intent_tokens(
        text: str,
    ) -> list[str]:
        """
        사용자가 직접 확정한 수치/날짜 표현만 추출한다.
        단순 숫자 전체를 비교하지 않고 단위가 있는 값만 대상으로 한다.
        """

        patterns = (
            r"20\d{2}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}\s*일?",
            r"\d+(?:\.\d+)?\s*%",
            r"\d[\d,]*\s*(?:원|만원|억원)",
            r"\d+\s*(?:개월|개월간|년|일|일간|회|회 이상|회 이하)",
        )

        tokens: list[str] = []

        for pattern in patterns:
            for match in re.findall(
                pattern,
                text,
            ):
                normalized = re.sub(
                    r"\s+",
                    "",
                    str(match),
                )

                if (
                    normalized
                    and normalized
                    not in tokens
                ):
                    tokens.append(
                        normalized
                    )

        return tokens

    @staticmethod
    def _contains_normalized_token(
        text: str,
        token: str,
    ) -> bool:
        normalized_text = re.sub(
            r"[\s,]",
            "",
            text,
        ).replace(
            "년",
            "-",
        ).replace(
            "월",
            "-",
        ).replace(
            "일",
            "",
        )

        normalized_token = re.sub(
            r"[\s,]",
            "",
            token,
        ).replace(
            "년",
            "-",
        ).replace(
            "월",
            "-",
        ).replace(
            "일",
            "",
        )

        return (
            normalized_token
            in normalized_text
        )

    @staticmethod
    def _is_keyword_negated(
        text: str,
        keyword: str,
    ) -> bool:
        keyword_index = (
            text.find(
                keyword
            )
        )

        if keyword_index < 0:
            return False

        start = max(
            0,
            keyword_index - 18,
        )

        end = min(
            len(text),
            keyword_index
            + len(keyword)
            + 24,
        )

        context = (
            text[
                start:end
            ]
        )

        negative_markers = (
            "없이",
            "없음",
            "없도록",
            "부과하지",
            "지급하지",
            "받지 않",
            "요구하지",
            "면제",
            "부담하지",
            "필요 없",
            "청구하지",
        )

        return any(
            marker in context
            for marker
            in negative_markers
        )

    def _collect_intent_value_tokens(
        self,
        message: str,
    ) -> list[str]:
        """
        기존 정규식 토큰 + 정규식이 놓치는 금액·퍼센트 표현
        ("5백만원", "1억 2천만원", "오만 원", "5.5퍼센트").
        이미 같은 값의 토큰이 있으면 추가하지 않는다.
        """

        tokens = self._extract_explicit_intent_tokens(
            message
        )

        covered_amounts: set[int] = set()
        covered_percents: set[float] = set()

        for token in tokens:
            covered_amounts |= find_amounts(token)
            covered_percents |= find_percents(token)

        for extra in extra_value_tokens(message):
            amounts = find_amounts(extra)
            percents = find_percents(extra)

            if amounts and amounts <= covered_amounts:
                continue

            if percents and percents <= covered_percents:
                continue

            if extra not in tokens:
                tokens.append(extra)
                covered_amounts |= amounts
                covered_percents |= percents

        return tokens

    @staticmethod
    def _is_reflected_date_fragment(
        *,
        token: str,
        message: str,
        clause: str,
    ) -> bool:
        """
        "2026년 12월 31일"에서 떨어져 나온 "2026년", "31일" 같은 조각은
        전체 날짜가 조항에 (표기만 달리) 들어 있으면 반영된 것으로 본다.
        """

        if not DATE_FRAGMENT_PATTERN.fullmatch(token):
            return False

        clause_dates = find_dates(clause)

        for match in DATE_PATTERN.finditer(message):
            phrase = re.sub(r"\s+", "", match.group(0))

            if token not in phrase:
                continue

            message_dates = find_dates(match.group(0))

            if message_dates and message_dates <= clause_dates:
                return True

        return False

    def _find_user_intent_conflicts(
        self,
        *,
        message: str,
        clause: str,
        contract: dict[str, Any] | None = None,
    ) -> list[str]:
        conflicts: list[str] = []

        for token in (
            self._collect_intent_value_tokens(
                message
            )
        ):
            if self._contains_normalized_token(
                clause,
                token,
            ):
                continue

            # 글자가 달라도 같은 값이면 반영된 것으로 본다.
            # (5만원 = 50,000원, 2026.12.31 = 2026년 12월 31일,
            #  오늘로부터 1년 뒤 = 계산된 날짜)
            if is_value_reflected(
                token=token,
                message=message,
                clause=clause,
                contract=contract,
            ):
                continue

            if self._is_reflected_date_fragment(
                token=token,
                message=message,
                clause=clause,
            ):
                continue

            conflicts.append(
                (
                    "사용자가 직접 지정한 조건 "
                    f"'{token}'이 생성 조항에 "
                    "그대로 반영되지 않았습니다."
                )
            )

        for keyword in (
            self.NEGATED_INTENT_KEYWORDS
        ):
            if keyword not in message:
                continue

            if (
                not self._is_keyword_negated(
                    message,
                    keyword,
                )
            ):
                continue

            if keyword not in clause:
                continue

            if (
                not self._is_keyword_negated(
                    clause,
                    keyword,
                )
            ):
                conflicts.append(
                    (
                        "사용자는 "
                        f"'{keyword}'을(를) 두지 않는 조건을 "
                        "요청했지만 생성 조항은 해당 의무/비용을 "
                        "부과할 가능성이 있습니다."
                    )
                )

        return (
            self._deduplicate_strings(
                conflicts
            )
        )

    def _requires_semantic_intent_review(
        self,
        *,
        message: str,
        conflicts: list[str],
    ) -> bool:
        """
        명백한 사용자 의도 충돌은 이미
        user_intent_conflict 사유로 LLM 검증 대상이 된다.

        여기서는 deterministic rule만으로
        의미 일치를 확정하기 어려운 제한/예외/순서 조건만
        추가 semantic review 대상으로 판단한다.

        예:
        - "3개월 연속 연체한 경우에만"
        - "원금에 대해서만"
        - "담보를 제외하고"
        - "오직 만기 이후에만"

        반면 아래와 같은 일반 요청은
        Slot 및 명시 조건 검증을 통과하면 Fast Path를 유지한다.

        - "연체했을 때 위약금을 정하고 싶어요"
        - "중도상환 조항을 넣고 싶어요"
        - "연체하면 지연손해금을 정하고 싶어요"

        "수수료 없이", "위약금 없이" 등의 부정 조건은
        _find_user_intent_conflicts()에서 별도로 검증한다.
        """

        if conflicts:
            return False

        normalized = (
            message.strip()
        )

        if not normalized:
            return False

        return any(
            marker in normalized
            for marker
            in self.SEMANTIC_INTENT_MARKERS
        )

    def _find_unresolved_ambiguous_phrases(
        self,
        *,
        clause: str,
        message: str,
    ) -> list[dict[str, str]]:
        found: list[
            dict[str, str]
        ] = []

        normalized_message = (
            message.strip()
        )

        for (
            phrase,
            reason,
        ) in self.AMBIGUOUS_PATTERNS:
            if phrase not in clause:
                continue

            if phrase in normalized_message:
                continue

            found.append(
                {
                    "phrase": phrase,
                    "reason": reason,
                }
            )

        return found

    def _find_legal_assertions(
        self,
        clause: str,
    ) -> list[str]:
        return [
            pattern
            for pattern
            in self.LEGAL_ASSERTION_PATTERNS
            if pattern in clause
        ]

    def _extract_slots(
        self,
        clause: str,
    ) -> list[str]:
        return [
            item.strip()
            for item
            in self.SLOT_PATTERN.findall(
                clause
            )
            if item.strip()
        ]

    def _find_missing_slots(
        self,
        *,
        unresolved_decisions: list[
            dict[str, str]
        ],
        existing_slots: list[str],
    ) -> list[str]:
        if not unresolved_decisions:
            return []

        normalized_slots = [
            self._normalize_label(
                slot
            )
            for slot in existing_slots
        ]

        missing: list[str] = []

        for decision in (
            unresolved_decisions
        ):
            name = str(
                decision.get(
                    "name",
                    "",
                )
            ).strip()

            if not name:
                continue

            normalized_name = (
                self._normalize_label(
                    name
                )
            )

            matched = any(
                normalized_name
                == slot
                or normalized_name
                in slot
                or slot
                in normalized_name
                for slot
                in normalized_slots
            )

            if not matched:
                missing.append(
                    name
                )

        return missing

    @staticmethod
    def _normalize_label(
        value: str,
    ) -> str:
        return re.sub(
            r"[\s·ㆍ,./()_\-]",
            "",
            value,
        ).lower()

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
    def _normalize_unresolved_decisions(
        value: Any,
    ) -> list[dict[str, str]]:
        if not isinstance(
            value,
            list,
        ):
            return []

        result: list[
            dict[str, str]
        ] = []

        for item in value:
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

            result.append(
                {
                    "name": name,
                    "reason": reason,
                }
            )

        return result

    # ==========================================
    # Domain Validation
    # ==========================================

    @staticmethod
    def _domain_consistency_check(
        domain_validation: dict[
            str,
            Any,
        ],
    ) -> dict[str, str]:
        consistency = (
            domain_validation.get(
                "consistencyCheck"
            )
        )

        if isinstance(
            consistency,
            dict,
        ):
            return {
                "type": (
                    "CONTRACT_CONSISTENCY"
                ),
                "status": str(
                    consistency.get(
                        "status",
                        "PASS",
                    )
                ).upper(),
                "message": str(
                    consistency.get(
                        "message",
                        "",
                    )
                ).strip(),
            }

        return {
            "type": (
                "CONTRACT_CONSISTENCY"
            ),
            "status": "PASS",
            "message": (
                "확정적으로 판단 가능한 "
                "계약 기본정보에서 "
                "직접적인 충돌은 발견되지 않았습니다."
            ),
        }

    # ==========================================
    # 공통 결과 생성
    # ==========================================

    def _build_result(
        self,
        *,
        suggested_clause: str,
        original_clause: str,
        warnings: list[str],
        status: str,
        is_safe: bool,
        checks: list[dict[str, str]],
        domain_validation: dict[
            str,
            Any,
        ],
        validator_mode: str,
        ambiguous_matches: list[
            dict[str, str]
        ],
        slots: list[str],
        missing_slots: list[str],
        legal_assertions: list[str],
        llm_reasons: list[str],
        unauthorized_additions: (
            list[str] | None
        ) = None,
    ) -> dict[str, Any]:
        normalized_status = (
            status.upper()
        )

        if normalized_status not in {
            "PASS",
            "REVIEW",
            "WARNING",
        }:
            normalized_status = (
                "REVIEW"
            )

        normalized_checks = (
            self._normalize_checks(
                checks
            )
        )

        normalized_warnings = (
            self._deduplicate_strings(
                warnings
            )
        )

        final_clause = str(
            suggested_clause
            or original_clause
            or ""
        ).strip()

        requires_revision = (
            bool(original_clause)
            and final_clause
            != original_clause
        )

        return {
            "revisedClause": (
                final_clause
            ),

            "suggestedClause": (
                final_clause
            ),

            "warnings": (
                normalized_warnings
            ),

            "status": (
                normalized_status
            ),

            "isSafeToSuggest": (
                bool(is_safe)
            ),

            "checks": (
                normalized_checks
            ),

            "requiresRevision": (
                requires_revision
            ),

            "unauthorizedAdditions": (
                unauthorized_additions
                or []
            ),

            "domainValidation": (
                domain_validation
            ),

            "validatorMode": (
                validator_mode
            ),

            "validation": {
                "status": (
                    normalized_status
                ),
                "isSafeToSuggest": (
                    bool(is_safe)
                ),
                "checks": (
                    normalized_checks
                ),
            },

            "debug": {
                "validatorMode": (
                    validator_mode
                ),
                "detectedAmbiguousPhrases": (
                    ambiguous_matches
                ),
                "slots": (
                    slots
                ),
                "missingSlots": (
                    missing_slots
                ),
                "legalAssertions": (
                    legal_assertions
                ),
                "llmReasons": (
                    llm_reasons
                ),
            },
        }

    # ==========================================
    # LLM Prompt
    # ==========================================

    def _build_system_prompt(
        self,
    ) -> str:
        return """
당신은 금전소비대차계약 조항의 최종 검증기다.

Python 기반 deterministic validator가 먼저 검증을 수행했고,
구조적 또는 법적 판단이 필요한 경우에만 당신에게 요청이 전달된다.

Generator가 만든 조항을 그대로 믿지 말고 다음을 검증한다.

1. GROUNDING
검색된 법령 근거와 모순되는 법적 단정이 있는지 확인한다.

2. USER_INTENT
사용자가 요구한 조건을 제대로 반영했는지 확인한다.
사용자가 요청하지 않은 새로운 의무나 제한을 임의로 추가하면 안 된다.

3. CONTRACT_CONSISTENCY
현재 계약의 원금, 변제기, 상환방식, 상환일 등과 충돌하는지 확인한다.

4. AMBIGUITY
실제 계약 이행에 필요한 중요한 조건이
결정되지 않은 채 애매한 표현으로 남았는지 확인한다.


[미결정 계약조건]

사용자 요청 또는 현재 계약정보에 값이 없는 중요한 계약조건을
임의로 결정해서는 안 된다.

반드시:

[입력 필요: 조건명]

형식으로 남긴다.


특히 아래 표현으로 결정을 회피하지 않는다.

- 전부 또는 일부
- 일부 또는 전부
- 별도로 정한다
- 별도로 합의한다
- 추후 합의한다
- 당사자가 정한다
- 당사자가 합의한다
- 상당한 기간
- 적절한 방법
- 필요한 경우

단, 사용자가 해당 범위나 표현을 직접 요청했다면
그 표현 자체가 확정된 조건일 수 있다.


[Slot]

반드시 정확히:

[입력 필요: 조건명]

형식을 사용한다.

한 Slot에는 하나의 결정값만 넣는다.


[법령]

검색 결과에 없는 법 조문이나 법률 내용을 만들어내지 않는다.

법적 근거가 충분하지 않으면 조항에 단정적으로 넣지 말고
warnings에 남긴다.


[출력]

반드시 JSON 객체 하나만 출력한다.

{
  "suggestedClause": "검증 후 최종 조항",
  "warnings": [],
  "unauthorizedAdditions": [],
  "validation": {
    "status": "PASS 또는 REVIEW 또는 WARNING",
    "isSafeToSuggest": true,
    "checks": [
      {
        "type": "GROUNDING",
        "status": "PASS 또는 REVIEW 또는 WARNING",
        "message": "검증 결과"
      },
      {
        "type": "USER_INTENT",
        "status": "PASS 또는 REVIEW 또는 WARNING",
        "message": "검증 결과"
      },
      {
        "type": "CONTRACT_CONSISTENCY",
        "status": "PASS 또는 REVIEW 또는 WARNING",
        "message": "검증 결과"
      },
      {
        "type": "AMBIGUITY",
        "status": "PASS 또는 REVIEW 또는 WARNING",
        "message": "검증 결과"
      }
    ]
  }
}
""".strip()

    def _build_user_prompt(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[dict[str, Any]],
        generated: dict[str, Any],
        references: list[dict[str, Any]],
        ambiguous_matches: list[
            dict[str, str]
        ],
        decision_candidates: list[str],
        existing_slots: list[str],
        missing_slots: list[str],
        domain_validation: dict[
            str,
            Any,
        ],
        llm_reasons: list[str],
        contract_clause_conflicts: list[str],
        user_intent_conflicts: list[str],
        semantic_intent_review: bool,
    ) -> str:
        reference_summaries = [
            self._reference_summary(
                reference
            )
            for reference
            in references
        ]

        return f"""
[사용자 요청]
{message}

[현재 계약정보]
{json.dumps(contract, ensure_ascii=False, indent=2)}

[법적 쟁점]
{json.dumps(issues, ensure_ascii=False, indent=2)}

[Generator 결과]
{json.dumps(generated, ensure_ascii=False, indent=2)}

[Python 계약 기본정보 검증]
{json.dumps(domain_validation, ensure_ascii=False, indent=2)}

[LLM 검증이 호출된 이유]
{json.dumps(llm_reasons, ensure_ascii=False, indent=2)}

[Python이 발견한 계약 확정값 충돌]
{json.dumps(contract_clause_conflicts, ensure_ascii=False, indent=2)}

[Python이 발견한 사용자 요청 충돌]
{json.dumps(user_intent_conflicts, ensure_ascii=False, indent=2)}

[자유문장 의미 검증 필요 여부]
{json.dumps(semantic_intent_review, ensure_ascii=False)}

[현재 생성 문장에서 발견된 모호 표현]
{json.dumps(ambiguous_matches, ensure_ascii=False, indent=2)}

[이미 존재하는 입력 Slot]
{json.dumps(existing_slots, ensure_ascii=False, indent=2)}

[Generator가 미결정이라고 표시했지만 Slot이 없는 조건]
{json.dumps(missing_slots, ensure_ascii=False, indent=2)}

[해당 쟁점의 결정사항 후보]
{json.dumps(decision_candidates, ensure_ascii=False, indent=2)}

후보를 전부 Slot으로 만들 필요는 없다.

다만 실제 계약의 권리·의무나 금액, 시점, 범위를 결정하는 데
필요한 조건인데 사용자와 현재 계약정보 어디에도 값이 없다면
반드시 [입력 필요: ...] 형태로 남긴다.

[검색 법령]
{json.dumps(reference_summaries, ensure_ascii=False, indent=2)}

위 정보만 이용해 최종 조항을 검증하라.
""".strip()

    @staticmethod
    def _reference_summary(
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = (
            reference.get(
                "metadata"
            )
            if isinstance(
                reference.get(
                    "metadata"
                ),
                dict,
            )
            else {}
        )

        return {
            "lawName": (
                reference.get(
                    "lawName"
                )
                or reference.get(
                    "law_name"
                )
                or metadata.get(
                    "law_name"
                )
            ),
            "articleNumber": (
                reference.get(
                    "articleNumber"
                )
                or reference.get(
                    "article_number"
                )
                or metadata.get(
                    "article_number"
                )
            ),
            "articleTitle": (
                reference.get(
                    "articleTitle"
                )
                or reference.get(
                    "article_title"
                )
                or metadata.get(
                    "article_title"
                )
            ),
            "content": (
                reference.get(
                    "content"
                )
                or reference.get(
                    "text"
                )
                or ""
            ),
        }

    # ==========================================
    # LLM 결과 정리
    # ==========================================

    def _normalize_llm_result(
        self,
        *,
        validated: dict[str, Any],
        generated: dict[str, Any],
        original_clause: str,
        domain_validation: dict[
            str,
            Any,
        ],
        ambiguous_matches: list[
            dict[str, str]
        ],
        existing_slots: list[str],
        missing_slots: list[str],
        legal_assertions: list[str],
        llm_reasons: list[str],
    ) -> dict[str, Any]:
        suggested_clause = str(
            validated.get(
                "suggestedClause"
            )
            or original_clause
        ).strip()

        warnings = (
            self._normalize_warnings(
                validated.get(
                    "warnings"
                )
            )
        )

        for warning in (
            self._normalize_warnings(
                generated.get(
                    "warnings"
                )
            )
        ):
            if warning not in warnings:
                warnings.append(
                    warning
                )

        validation = (
            validated.get(
                "validation"
            )
        )

        if not isinstance(
            validation,
            dict,
        ):
            validation = {}

        status = str(
            validation.get(
                "status"
            )
            or "REVIEW"
        ).upper()

        if status not in {
            "PASS",
            "REVIEW",
            "WARNING",
        }:
            status = (
                "REVIEW"
            )

        is_safe = bool(
            validation.get(
                "isSafeToSuggest",
                status != "WARNING",
            )
        )

        checks = (
            validation.get(
                "checks"
            )
        )

        if not isinstance(
            checks,
            list,
        ):
            checks = []

        normalized_checks = (
            self._normalize_checks(
                checks
            )
        )

        if not any(
            check.get(
                "type"
            )
            == "CONTRACT_CONSISTENCY"
            for check
            in normalized_checks
        ):
            normalized_checks.append(
                self._domain_consistency_check(
                    domain_validation
                )
            )

        remaining_ambiguity = (
            self._find_unresolved_ambiguous_phrases(
                clause=(
                    suggested_clause
                ),
                message="",
            )
        )

        final_slots = (
            self._extract_slots(
                suggested_clause
            )
        )

        if (
            remaining_ambiguity
            and not final_slots
        ):
            status = (
                "REVIEW"
            )

            warning = (
                "계약 문구에 아직 범위나 기준이 "
                "확정되지 않은 표현이 남아 있어 "
                "추가 확인이 필요합니다."
            )

            if warning not in warnings:
                warnings.append(
                    warning
                )

        unauthorized_additions = (
            validated.get(
                "unauthorizedAdditions"
            )
        )

        if not isinstance(
            unauthorized_additions,
            list,
        ):
            unauthorized_additions = []

        return self._build_result(
            suggested_clause=(
                suggested_clause
            ),
            original_clause=(
                original_clause
            ),
            warnings=warnings,
            status=status,
            is_safe=is_safe,
            checks=(
                normalized_checks
            ),
            domain_validation=(
                domain_validation
            ),
            validator_mode=(
                "LLM"
            ),
            ambiguous_matches=(
                ambiguous_matches
            ),
            slots=(
                final_slots
            ),
            missing_slots=(
                missing_slots
            ),
            legal_assertions=(
                legal_assertions
            ),
            llm_reasons=(
                llm_reasons
            ),
            unauthorized_additions=[
                str(
                    item
                ).strip()
                for item
                in unauthorized_additions
                if str(
                    item
                ).strip()
            ],
        )

    @staticmethod
    def _normalize_checks(
        checks: Any,
    ) -> list[dict[str, str]]:
        if not isinstance(
            checks,
            list,
        ):
            return []

        allowed_types = {
            "GROUNDING",
            "USER_INTENT",
            "CONTRACT_CONSISTENCY",
            "AMBIGUITY",
        }

        normalized: list[
            dict[str, str]
        ] = []

        for check in checks:
            if not isinstance(
                check,
                dict,
            ):
                continue

            check_type = str(
                check.get(
                    "type",
                    "",
                )
            ).upper()

            if (
                check_type
                not in allowed_types
            ):
                continue

            status = str(
                check.get(
                    "status",
                    "REVIEW",
                )
            ).upper()

            if status not in {
                "PASS",
                "REVIEW",
                "WARNING",
            }:
                status = (
                    "REVIEW"
                )

            normalized.append(
                {
                    "type": (
                        check_type
                    ),
                    "status": (
                        status
                    ),
                    "message": str(
                        check.get(
                            "message",
                            "",
                        )
                    ).strip(),
                }
            )

        return normalized

    @staticmethod
    def _deduplicate_strings(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            text = str(
                value
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

    # ==========================================
    # Issue별 결정 후보
    # ==========================================

    def _collect_decision_candidates(
        self,
        issue_types: list[str],
    ) -> list[str]:
        result: list[str] = []

        for issue_type in (
            issue_types
        ):
            for value in (
                self.ISSUE_REQUIRED_DECISIONS.get(
                    issue_type,
                    [],
                )
            ):
                if value not in result:
                    result.append(
                        value
                    )

        return result

    # ==========================================
    # OpenAI 응답 처리
    # ==========================================

    @staticmethod
    def _extract_text(
        response: Any,
    ) -> str:
        output_text = getattr(
            response,
            "output_text",
            None,
        )

        if output_text:
            return str(
                output_text
            ).strip()

        try:
            values: list[str] = []

            for item in (
                response.output
            ):
                for content in (
                    item.content
                ):
                    text = getattr(
                        content,
                        "text",
                        None,
                    )

                    if text:
                        values.append(
                            str(text)
                        )

            if values:
                return "\n".join(
                    values
                ).strip()

        except Exception:
            pass

        raise ValueError(
            "ClauseValidator 응답에서 "
            "텍스트를 찾지 못했습니다."
        )

    @staticmethod
    def _parse_json(
        raw_text: str,
    ) -> dict[str, Any]:
        cleaned = (
            raw_text
            .strip()
            .removeprefix(
                "```json"
            )
            .removeprefix(
                "```"
            )
            .removesuffix(
                "```"
            )
            .strip()
        )

        try:
            parsed = (
                json.loads(
                    cleaned
                )
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "ClauseValidator가 JSON 형식이 "
                "아닌 응답을 반환했습니다."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "ClauseValidator 응답은 "
                "JSON 객체여야 합니다."
            )

        return parsed