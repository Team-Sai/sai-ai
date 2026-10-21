from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ContractReviewRequest(
    BaseModel
):
    """
    기존 차용증 법률 검토 요청.
    """

    model_config = ConfigDict(
        extra="allow"
    )

    principal: float | None = None

    interestRate: float | None = None

    lateInterestRate: (
        float | None
    ) = None

    repaymentDate: (
        str | None
    ) = None


class ContractClauseContext(
    BaseModel
):
    """
    AI 특약 생성 시 전달되는
    현재 차용증의 기본정보.
    """

    model_config = ConfigDict(
        extra="allow"
    )

    principal: float | None = None

    interestRate: (
        float | None
    ) = None

    lateInterestRate: (
        float | None
    ) = None

    startDate: str | None = None

    repaymentDate: (
        str | None
    ) = None

    repaymentDay: (
        int | None
    ) = None

    repaymentType: (
        str | None
    ) = None

    contractAlias: (
        str | None
    ) = None

    terms: str | None = None


class ContractClauseSuggestionRequest(
    BaseModel
):
    """
    AI 특약 작성 Agent 요청.
    """

    contract: (
        ContractClauseContext
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )


class ContractReviewOptions(
    BaseModel
):
    topK: int = 3
    generate: bool = True