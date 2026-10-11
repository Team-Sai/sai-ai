from typing import Any


class ContractReviewQueryBuilder:
    """
    계약 입력값 중 실제 법률 근거 검색이 필요한 항목만
    Retrieval Query로 변환한다.

    모든 입력값을 무조건 검색하지 않는다.
    """

    def build_queries(
        self,
        contract: dict[str, Any],
    ) -> list[dict[str, str]]:

        queries = []

        principal = contract.get(
            "principal"
        )

        interest_rate = contract.get(
            "interestRate"
        )

        late_interest_rate = contract.get(
            "lateInterestRate"
        )

        repayment_date = contract.get(
            "repaymentDate"
        )

        # 약정 이자율
        if interest_rate is not None:

            principal_context = ""

            if principal is not None:
                principal_context = (
                    f" 대여원금 {principal}원"
                )

            queries.append(
                {
                    "field": "interestRate",
                    "type": "INTEREST_RATE",
                    "query": (
                        "개인 간 금전대차 계약 "
                        f"약정이자율 연 {interest_rate}%"
                        f"{principal_context} "
                        "이자제한법 최고이자율 "
                        "최고한도 초과이자 무효 "
                        "초과 지급 이자 원본 충당"
                    ),
                }
            )

        # 연체이자율 / 지연손해금
        if late_interest_rate is not None:

            queries.append(
                {
                    "field": "lateInterestRate",
                    "type": "LATE_INTEREST",
                    "query": (
                        "개인 간 금전대차 "
                        f"연체이자율 연 {late_interest_rate}% "
                        "금전채무 불이행 손해배상액 "
                        "지연손해금 약정이율 "
                        "배상액 예정 위약금 "
                        "부당하게 과다한 손해배상액 감액 "
                        "약정이율 법령 제한 "
                        "최고이자율 제한"
                    ),
                }
            )

        # 상환일이 없는 경우에만 관련 법률 검색
        if not repayment_date:

            queries.append(
                {
                    "field": "repaymentDate",
                    "type": "REPAYMENT_DATE",
                    "query": (
                        "금전소비대차 "
                        "반환시기 약정 없는 경우 "
                        "변제기 상환기일 "
                        "대주 반환 최고 "
                        "차용금 반환"
                    ),
                }
            )

        return queries