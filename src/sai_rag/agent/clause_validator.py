from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from sai_rag.agent.contract_domain_validator import (
    ContractDomainValidator,
)


load_dotenv()


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
    ) -> dict[str, Any]:
        """
        Generator 결과를 검증한다.

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
                        "분류된 계약 쟁점 범위 안에서 "
                        "조항이 생성되었습니다."
                    ),
                },
                self._domain_consistency_check(
                    domain_validation
                ),
                {
                    "type": "AMBIGUITY",
                    "status": "PASS",
                    "message": (
                        "미정 계약조건은 입력 Slot으로 "
                        "구조화되어 있으며 "
                        "추가적인 모호 표현이 "
                        "감지되지 않았습니다."
                    ),
                },
            ]

            return self._build_result(
                suggested_clause=(
                    original_clause
                ),
                original_clause=(
                    original_clause
                ),
                warnings=(
                    generator_warnings
                ),
                status="PASS",
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
            )
        )

        response = (
            self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
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

        if ambiguous_matches:
            reasons.append(
                "unresolved_ambiguity"
            )

        if missing_slots:
            reasons.append(
                "missing_required_slot"
            )

        if legal_assertions:
            reasons.append(
                "legal_assertion"
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

            # 사용자가 해당 범위를 직접 요청했다면
            # 그 표현 자체가 확정된 의도일 수 있다.
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