from typing import Any


class ContractRuleEngine:
    """
    명확하게 코드로 판정할 수 있는 규칙만 검사한다.

    원칙:
    - PASS: 실제 규칙으로 검사한 뒤 통과
    - WARNING: 규칙 위반 가능성 또는 수정 필요
    - ERROR: 입력값 자체가 잘못됨
    - REVIEW: 법률 근거 확인이 필요한 항목

    법률 해석이 필요한 내용을 임의로 PASS 처리하지 않는다.
    """

    MAX_INTEREST_RATE = 20.0

    def evaluate(
        self,
        contract: dict[str, Any],
    ) -> list[dict[str, Any]]:

        results = []

        self._evaluate_principal(
            contract=contract,
            results=results,
        )

        self._evaluate_interest_rate(
            contract=contract,
            results=results,
        )

        self._evaluate_late_interest_rate(
            contract=contract,
            results=results,
        )

        return results

    def _evaluate_principal(
        self,
        contract: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:

        principal = contract.get(
            "principal"
        )

        if principal is None:
            results.append(
                {
                    "field": "principal",
                    "status": "ERROR",
                    "code": "MISSING_PRINCIPAL",
                    "title": "대여금액을 입력해주세요.",
                    "message": (
                        "차용증 작성을 위해 "
                        "대여금액이 필요합니다."
                    ),
                    "scope": "INPUT",
                }
            )

            return

        if principal <= 0:
            results.append(
                {
                    "field": "principal",
                    "status": "ERROR",
                    "code": "INVALID_PRINCIPAL",
                    "title": "대여금액을 확인해주세요.",
                    "message": (
                        "대여금액은 0보다 "
                        "커야 합니다."
                    ),
                    "scope": "INPUT",
                }
            )

            return

        results.append(
            {
                "field": "principal",
                "status": "PASS",
                "code": "VALID_PRINCIPAL",
                "title": "대여금액 입력 확인",
                "message": (
                    "대여금액 입력값이 "
                    "유효합니다."
                ),
                "scope": "INPUT",
            }
        )

    def _evaluate_interest_rate(
        self,
        contract: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:

        interest_rate = contract.get(
            "interestRate"
        )

        if interest_rate is None:
            return

        if interest_rate < 0:
            results.append(
                {
                    "field": "interestRate",
                    "status": "ERROR",
                    "code": "INVALID_INTEREST_RATE",
                    "title": "이자율을 확인해주세요.",
                    "message": (
                        "이자율은 0 이상이어야 합니다."
                    ),
                    "scope": "LEGAL",
                }
            )

            return

        if (
            interest_rate
            > self.MAX_INTEREST_RATE
        ):
            results.append(
                {
                    "field": "interestRate",
                    "status": "WARNING",
                    "code": "INTEREST_RATE_EXCEEDED",
                    "title": (
                        "약정 이자율을 "
                        "확인해주세요."
                    ),
                    "message": (
                        f"입력한 연 {interest_rate}%는 "
                        f"현재 기준값 "
                        f"연 {self.MAX_INTEREST_RATE}%를 "
                        "초과합니다."
                    ),
                    "scope": "LEGAL",
                }
            )

            return

        results.append(
            {
                "field": "interestRate",
                "status": "PASS",
                "code": "INTEREST_RATE_WITHIN_LIMIT",
                "title": "약정 이자율 기준 확인",
                "message": (
                    f"입력한 연 {interest_rate}%는 "
                    f"현재 기준값 "
                    f"연 {self.MAX_INTEREST_RATE}% "
                    "이하입니다."
                ),
                "scope": "LEGAL",
            }
        )

    @staticmethod
    def _evaluate_late_interest_rate(
        contract: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> None:

        late_interest_rate = contract.get(
            "lateInterestRate"
        )

        if late_interest_rate is None:
            return

        if late_interest_rate < 0:
            results.append(
                {
                    "field": "lateInterestRate",
                    "status": "ERROR",
                    "code": (
                        "INVALID_LATE_INTEREST_RATE"
                    ),
                    "title": (
                        "연체이자율을 확인해주세요."
                    ),
                    "message": (
                        "연체이자율은 "
                        "0 이상이어야 합니다."
                    ),
                    "scope": "LEGAL",
                }
            )

            return

        results.append(
            {
                "field": "lateInterestRate",
                "status": "REVIEW",
                "code": "LATE_INTEREST_REVIEW",
                "title": (
                    "연체이자율 법률 검토"
                ),
                "message": (
                    "입력값은 유효하지만 "
                    "적용 가능한 법률 기준과 "
                    "손해배상 관련 규정을 "
                    "추가로 확인합니다."
                ),
                "scope": "LEGAL",
            }
        )