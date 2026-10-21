from typing import Any


class ContractContextFormatter:
    """
    내부 계약 DTO의 필드명과 값을
    LLM이 오해하지 않도록 업무 의미가 포함된
    자연어 Context로 변환한다.

    예:
    repaymentDate
        → 계약의 최종 만기일

    repaymentDay
        → 분할상환 시 매월 상환하기로 한 날짜

    두 값은 서로 다른 개념이므로
    날짜가 다르다는 이유만으로 충돌로 판단하면 안 된다.
    """

    REPAYMENT_TYPE_LABELS = {
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
            "만기일시상환"
        ),
    }

    @classmethod
    def format(
        cls,
        contract: dict[str, Any],
    ) -> str:
        lines: list[str] = []

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

        contract_alias = contract.get(
            "contractAlias"
        )

        terms = contract.get(
            "terms"
        )

        lines.append(
            "[계약 필드 의미]"
        )

        lines.append(
            "- principal: 대여하기로 약정한 원금"
        )

        lines.append(
            "- interestRate: 원금에 적용하는 연 약정이자율(%)"
        )

        lines.append(
            "- lateInterestRate: 연체 발생 시 적용하려는 연체이자율(%)"
        )

        lines.append(
            "- startDate: 대여 및 계약이 시작되는 날짜"
        )

        lines.append(
            "- repaymentDate: 계약 전체의 최종 만기일 또는 최종 변제기"
        )

        lines.append(
            "- repaymentDay: 분할상환 계약에서 매월 상환하기로 한 날짜"
        )

        lines.append(
            "- repaymentType: 원금을 어떤 방식으로 나누어 상환하는지 나타내는 방식"
        )

        lines.append(
            "- contractAlias: 사용자가 계약을 구분하기 위해 작성한 계약 목적 또는 별칭"
        )

        lines.append(
            "- terms: 이미 계약서에 작성되어 있는 추가 특약사항"
        )

        lines.append("")

        lines.append(
            "[중요한 해석 규칙]"
        )

        lines.append(
            "- repaymentDate와 repaymentDay는 서로 다른 개념이다."
        )

        lines.append(
            "- repaymentDate의 날짜와 repaymentDay의 숫자가 다르다는 이유만으로 계약정보 충돌이라고 판단하지 않는다."
        )

        lines.append(
            "- 예를 들어 repaymentDate가 2027-10-09이고 repaymentDay가 10이라면, "
            "이는 '최종 만기일은 2027년 10월 9일이고 정기 분할상환일은 매월 10일'이라는 의미일 수 있다."
        )

        lines.append(
            "- 실제 계약상 충돌 여부는 상환방식, 최초 상환일, 회차별 일정 등 추가 정보가 있을 때 판단한다."
        )

        lines.append("")

        lines.append(
            "[현재 계약 값]"
        )

        if principal is not None:
            lines.append(
                f"- 대여원금: {principal}원"
            )
        else:
            lines.append(
                "- 대여원금: 미입력"
            )

        if interest_rate is not None:
            lines.append(
                f"- 약정이자율: 연 {interest_rate}%"
            )
        else:
            lines.append(
                "- 약정이자율: 미입력"
            )

        if late_interest_rate is not None:
            lines.append(
                f"- 연체이자율: 연 {late_interest_rate}%"
            )
        else:
            lines.append(
                "- 연체이자율: 별도 입력 없음"
            )

        if start_date:
            lines.append(
                f"- 계약 시작일: {start_date}"
            )
        else:
            lines.append(
                "- 계약 시작일: 미입력"
            )

        if repayment_date:
            lines.append(
                f"- 최종 만기일: {repayment_date}"
            )
        else:
            lines.append(
                "- 최종 만기일: 미입력"
            )

        if repayment_day is not None:
            lines.append(
                f"- 매월 분할상환일: 매월 {repayment_day}일"
            )
        else:
            lines.append(
                "- 매월 분할상환일: 미입력"
            )

        if repayment_type:
            repayment_label = (
                cls.REPAYMENT_TYPE_LABELS.get(
                    str(repayment_type),
                    str(repayment_type),
                )
            )

            lines.append(
                f"- 상환방식: {repayment_label} "
                f"({repayment_type})"
            )
        else:
            lines.append(
                "- 상환방식: 미입력"
            )

        if contract_alias:
            lines.append(
                f"- 계약 목적/별칭: {contract_alias}"
            )

        if terms:
            lines.append(
                f"- 기존 특약사항: {terms}"
            )
        else:
            lines.append(
                "- 기존 특약사항: 없음"
            )

        return "\n".join(
            lines
        )