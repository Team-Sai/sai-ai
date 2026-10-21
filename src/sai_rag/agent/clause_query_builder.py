from typing import Any


class ClauseQueryBuilder:
    """
    법률 쟁점 분석 결과를 기반으로
    쟁점별 검색 Query와 법령 검색 범위를 만든다.
    """

    ISSUE_LAW_SCOPE: dict[
        str,
        list[str],
    ] = {
        "REPAYMENT": [
            "민법",
        ],

        "DEFAULT": [
            "민법",
        ],

        "ACCELERATION": [
            "민법",
        ],

        "INTEREST": [
            "이자제한법",
            "이자제한법 제2조제1항의 최고이자율에 관한 규정",
            "민법",
        ],

        "LATE_INTEREST": [
            "민법",
            "이자제한법",
            "이자제한법 제2조제1항의 최고이자율에 관한 규정",
        ],

        "PENALTY": [
            "민법",
        ],

        "EARLY_REPAYMENT": [
            "민법",
        ],

        "GUARANTEE": [
            "민법",
        ],

        "COLLATERAL": [
            "민법",
        ],

        "EXECUTION": [
            "민사집행법",
            "민법",
        ],

        "ELECTRONIC_SIGNATURE": [
            "전자서명법",
            "전자문서 및 전자거래 기본법",
        ],

        "OTHER": [
            "민법",
        ],
    }

    ISSUE_BASE_TERMS: dict[
        str,
        list[str],
    ] = {
        "REPAYMENT": [
            "금전소비대차",
            "변제기",
            "분할상환",
            "변제",
            "상환기일",
        ],

        "DEFAULT": [
            "채무불이행",
            "이행지체",
            "상환기일 미지급",
            "금전채무",
        ],

        "ACCELERATION": [
            "잔존 채무 일시 변제",
            "기한의 이익",
            "분할상환 연체",
            "잔여 원금",
        ],

        "INTEREST": [
            "금전대차 이자",
            "약정이율",
            "최고이자율",
        ],

        "LATE_INTEREST": [
            "금전채무불이행",
            "지연손해금",
            "연체이자",
            "약정이율",
        ],

        "PENALTY": [
            "손해배상액 예정",
            "위약금",
            "과다한 배상액",
            "감액",
        ],

        "EARLY_REPAYMENT": [
            "만기 전 변제",
            "중도상환",
            "채무 변제",
        ],

        "GUARANTEE": [
            "보증채무",
            "연대보증",
            "보증인",
        ],

        "COLLATERAL": [
            "담보",
            "담보 제공",
            "담보권",
        ],

        "EXECUTION": [
            "강제집행",
            "집행권원",
            "채권 집행",
        ],

        "ELECTRONIC_SIGNATURE": [
            "전자서명",
            "전자문서",
            "전자계약",
        ],

        "OTHER": [
            "금전소비대차 계약",
            "채권자 채무자 권리 의무",
        ],
    }

    def build_queries(
        self,
        contract: dict[str, Any],
        message: str,
        issues: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        if not message.strip():
            raise ValueError(
                "특약 생성을 위한 사용자 요청이 비어 있습니다."
            )

        contract_context = (
            self._build_contract_context(
                contract
            )
        )

        search_plans: list[
            dict[str, Any]
        ] = []

        for issue in issues:
            issue_type = str(
                issue.get(
                    "type",
                    "OTHER",
                )
            )

            analyzer_terms = issue.get(
                "searchTerms",
                [],
            )

            base_terms = (
                self.ISSUE_BASE_TERMS.get(
                    issue_type,
                    self.ISSUE_BASE_TERMS[
                        "OTHER"
                    ],
                )
            )

            law_names = (
                self.ISSUE_LAW_SCOPE.get(
                    issue_type,
                    self.ISSUE_LAW_SCOPE[
                        "OTHER"
                    ],
                )
            )

            all_terms = self._unique_terms(
                [
                    *base_terms,
                    *analyzer_terms,
                ]
            )

            primary_query = " ".join(
                [
                    *all_terms,
                    contract_context,
                ]
            ).strip()

            contextual_query = " ".join(
                [
                    message.strip(),
                    *all_terms,
                ]
            ).strip()

            search_plans.append(
                {
                    "issueType": issue_type,
                    "query": primary_query,
                    "lawNames": law_names,
                }
            )

            search_plans.append(
                {
                    "issueType": issue_type,
                    "query": contextual_query,
                    "lawNames": law_names,
                }
            )

        return search_plans

    def _build_contract_context(
        self,
        contract: dict[str, Any],
    ) -> str:
        parts: list[str] = []

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

        repayment_type = contract.get(
            "repaymentType"
        )

        repayment_day = contract.get(
            "repaymentDay"
        )

        if principal is not None:
            parts.append(
                f"대여원금 {principal}원"
            )

        if interest_rate is not None:
            parts.append(
                f"약정이율 연 {interest_rate}%"
            )

        if late_interest_rate is not None:
            parts.append(
                f"연체이율 연 {late_interest_rate}%"
            )

        if repayment_date:
            parts.append(
                f"변제기 {repayment_date}"
            )

        if repayment_type:
            parts.append(
                f"상환방식 {repayment_type}"
            )

        if repayment_day is not None:
            parts.append(
                f"매월 상환일 {repayment_day}일"
            )

        return " ".join(
            parts
        )

    def _unique_terms(
        self,
        terms: list[str],
    ) -> list[str]:
        result: list[str] = []

        seen: set[str] = set()

        for term in terms:
            normalized = str(
                term
            ).strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                normalized
            )

        return result