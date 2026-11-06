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

class ClauseChatRouteRequest(
    BaseModel
):
    """
    대화형 특약: 사용자 답장 분류 요청.
    대화 상태(값 칸, 질문, 답)는 백엔드가 보관하다가 같이 보낸다.
    """

    contract: (
        ContractClauseContext
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    slots: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    questions: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    answers: dict[str, str] = Field(
        default_factory=dict,
    )

class ClauseChatWriteRequest(
    BaseModel
):
    """
    대화형 특약: 확정한 조건으로 특약 쓰기 요청.
    issues는 /understand가 돌려준 값을 그대로 보낸다.
    """

    contract: (
        ContractClauseContext
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    issues: list[dict[str, Any]] = Field(
        min_length=1,
    )

    slots: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    questions: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    answers: dict[str, str] = Field(
        default_factory=dict,
    )


    # /prepare로 미리 찾은 근거 법령. 백엔드가 보관했다가 보낸다. 없으면 지금처럼 검색한다.
    references: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=10,
    )


class ClauseChatPrepareRequest(
    BaseModel
):
    """
    대화형 특약: 사용자가 답하는 동안 근거 법령을 미리 찾는 요청.
    issues는 /understand가 돌려준 값을 그대로 보낸다.
    """

    contract: (
        ContractClauseContext
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    issues: list[dict[str, Any]] = Field(
        min_length=1,
    )


class ClauseReviewRequest(
    BaseModel
):
    """
    반영한 AI 조항 다시 검토 요청.
    사용자가 [입력 필요] 칸을 채우거나 문장을 고친 뒤 보낸다.
    """

    contract: (
        ContractClauseContext
    )

    message: str = Field(
        min_length=1,
        max_length=2000,
    )

    issues: list[dict[str, Any]] = Field(
        min_length=1,
    )

    clause: str = Field(
        min_length=1,
        max_length=5000,
    )

    references: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=10,
    )



class ContractReviewOptions(
    BaseModel
):
    topK: int = 3
    generate: bool = True