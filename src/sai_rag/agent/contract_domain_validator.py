from datetime import date
from typing import Any


class ContractDomainValidator:
    """
    LLM에게 맡길 필요가 없는
    확정적인 계약 데이터 규칙을 Python 코드로 검증한다.

    핵심 원칙:
    - deterministic rule은 코드에서 판단
    - 법률 해석/문맥 판단만 LLM에게 맡긴다

    특히:
    repaymentDate = 계약 전체의 최종 만기일
    repaymentDay  = 매월 분할상환일

    두 값은 서로 다른 개념이므로
    서로 숫자/날짜가 다르다는 이유로 충돌하지 않는다.
    """

    SUPPORTED_REPAYMENT_TYPES = {
        "EQUAL_PRINCIPAL_AND_INTEREST",
        "EQUAL_PRINCIPAL",
        "BULLET",
        "LUMP_SUM",
    }

    def validate(
        self,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        checks: list[
            dict[str, str]
        ] = []

        facts: list[str] = []

        principal = contract.get(
            "principal"
        )

        interest_rate = contract.get(
            "interestRate"
        )

        late_interest_rate = contract.get(
            "lateInterestRate"
        )

        start_date = contract.get(
            "startDate"
        )

        repayment_date = contract.get(
            "repaymentDate"
        )

        repayment_day = contract.get(
            "repaymentDay"
        )

        repayment_type = contract.get(
            "repaymentType"
        )

        # ======================================
        # 원금
        # ======================================

        if principal is not None:
            try:
                principal_value = float(
                    principal
                )

                if principal_value <= 0:
                    checks.append(
                        {
                            "type": "PRINCIPAL",
                            "status": "WARNING",
                            "message": (
                                "대여원금은 0보다 커야 합니다."
                            ),
                        }
                    )
                else:
                    facts.append(
                        f"대여원금은 {principal_value}원이다."
                    )

            except (
                TypeError,
                ValueError,
            ):
                checks.append(
                    {
                        "type": "PRINCIPAL",
                        "status": "WARNING",
                        "message": (
                            "대여원금 값을 숫자로 해석할 수 없습니다."
                        ),
                    }
                )

        # ======================================
        # 약정이자율
        # ======================================

        if interest_rate is not None:
            try:
                rate = float(
                    interest_rate
                )

                if rate < 0:
                    checks.append(
                        {
                            "type": "INTEREST_RATE",
                            "status": "WARNING",
                            "message": (
                                "약정이자율은 음수가 될 수 없습니다."
                            ),
                        }
                    )
                else:
                    facts.append(
                        f"약정이자율은 연 {rate}%이다."
                    )

            except (
                TypeError,
                ValueError,
            ):
                checks.append(
                    {
                        "type": "INTEREST_RATE",
                        "status": "WARNING",
                        "message": (
                            "약정이자율 값을 숫자로 해석할 수 없습니다."
                        ),
                    }
                )

        # ======================================
        # 연체이자율
        # ======================================

        if late_interest_rate is not None:
            try:
                late_rate = float(
                    late_interest_rate
                )

                if late_rate < 0:
                    checks.append(
                        {
                            "type": "LATE_INTEREST_RATE",
                            "status": "WARNING",
                            "message": (
                                "연체이자율은 음수가 될 수 없습니다."
                            ),
                        }
                    )
                else:
                    facts.append(
                        f"연체이자율은 연 {late_rate}%이다."
                    )

            except (
                TypeError,
                ValueError,
            ):
                checks.append(
                    {
                        "type": "LATE_INTEREST_RATE",
                        "status": "WARNING",
                        "message": (
                            "연체이자율 값을 숫자로 해석할 수 없습니다."
                        ),
                    }
                )

        # ======================================
        # 날짜
        # ======================================

        parsed_start_date = (
            self._parse_date(
                start_date
            )
        )

        parsed_repayment_date = (
            self._parse_date(
                repayment_date
            )
        )

        if start_date:
            if parsed_start_date:
                facts.append(
                    f"계약 시작일은 {start_date}이다."
                )
            else:
                checks.append(
                    {
                        "type": "START_DATE",
                        "status": "WARNING",
                        "message": (
                            "계약 시작일의 날짜 형식이 올바르지 않습니다."
                        ),
                    }
                )

        if repayment_date:
            if parsed_repayment_date:
                facts.append(
                    (
                        "repaymentDate는 계약 전체의 "
                        f"최종 만기일이며 값은 {repayment_date}이다."
                    )
                )
            else:
                checks.append(
                    {
                        "type": "REPAYMENT_DATE",
                        "status": "WARNING",
                        "message": (
                            "최종 만기일의 날짜 형식이 올바르지 않습니다."
                        ),
                    }
                )

        if (
            parsed_start_date
            and parsed_repayment_date
        ):
            if (
                parsed_repayment_date
                <= parsed_start_date
            ):
                checks.append(
                    {
                        "type": "DATE_ORDER",
                        "status": "WARNING",
                        "message": (
                            "최종 만기일은 계약 시작일 이후여야 합니다."
                        ),
                    }
                )
            else:
                facts.append(
                    (
                        "최종 만기일은 계약 시작일 이후이므로 "
                        "날짜 순서상 충돌이 없다."
                    )
                )

        # ======================================
        # 매월 상환일
        # ======================================

        if repayment_day is not None:
            try:
                day = int(
                    repayment_day
                )

                if (
                    day < 1
                    or day > 31
                ):
                    checks.append(
                        {
                            "type": "REPAYMENT_DAY",
                            "status": "WARNING",
                            "message": (
                                "매월 분할상환일은 1일부터 31일 사이여야 합니다."
                            ),
                        }
                    )

                else:
                    facts.append(
                        (
                            "repaymentDay는 최종 만기일이 아니라 "
                            f"매월 분할상환일이며 값은 매월 {day}일이다."
                        )
                    )

            except (
                TypeError,
                ValueError,
            ):
                checks.append(
                    {
                        "type": "REPAYMENT_DAY",
                        "status": "WARNING",
                        "message": (
                            "매월 분할상환일 값을 정수로 해석할 수 없습니다."
                        ),
                    }
                )

        # ======================================
        # repaymentDate ↔ repaymentDay
        #
        # 중요:
        # 둘은 비교 대상이 아니다.
        # ======================================

        if (
            repayment_date
            and repayment_day is not None
        ):
            facts.append(
                (
                    "최종 만기일(repaymentDate)과 "
                    "매월 분할상환일(repaymentDay)은 "
                    "서로 다른 역할의 필드이므로 "
                    "값이 다르다는 이유만으로 충돌하지 않는다."
                )
            )

        # ======================================
        # 상환방식
        # ======================================

        if repayment_type:
            repayment_type_text = str(
                repayment_type
            )

            if (
                repayment_type_text
                not in self.SUPPORTED_REPAYMENT_TYPES
            ):
                checks.append(
                    {
                        "type": "REPAYMENT_TYPE",
                        "status": "REVIEW",
                        "message": (
                            "현재 시스템에서 명확히 해석하지 못하는 "
                            f"상환방식입니다: {repayment_type_text}"
                        ),
                    }
                )
            else:
                facts.append(
                    (
                        "상환방식은 "
                        f"{repayment_type_text}이다."
                    )
                )

        # ======================================
        # 최종 CONTRACT_CONSISTENCY
        # ======================================

        blocking_checks = [
            check
            for check in checks
            if check.get(
                "status"
            ) == "WARNING"
        ]

        if blocking_checks:
            consistency_status = (
                "WARNING"
            )

            consistency_message = (
                "계약 기본정보 중 코드 기반 검증에서 "
                "확인이 필요한 값이 있습니다."
            )
        else:
            consistency_status = (
                "PASS"
            )

            consistency_message = (
                "확정적으로 판단 가능한 계약 기본정보에서 "
                "직접적인 충돌은 발견되지 않았습니다."
            )

        consistency_check = {
            "type": (
                "CONTRACT_CONSISTENCY"
            ),
            "status": (
                consistency_status
            ),
            "message": (
                consistency_message
            ),
        }

        return {
            "status": (
                consistency_status
            ),

            "facts": (
                facts
            ),

            "fieldChecks": (
                checks
            ),

            "consistencyCheck": (
                consistency_check
            ),
        }

    def _parse_date(
        self,
        value: Any,
    ) -> date | None:
        if not value:
            return None

        try:
            return date.fromisoformat(
                str(value)
            )
        except ValueError:
            return None