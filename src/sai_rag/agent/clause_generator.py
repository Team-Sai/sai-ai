from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from sai_rag.agent.openai_options import (
    openai_call_options,
)


load_dotenv()


class ClauseGenerator:
    """
    계약 조항 초안을 생성한다.

    V5.4

    핵심 원칙:
    1. 검색된 법령 근거 안에서만 법적 설명을 작성한다.
    2. 사용자가 이미 정한 조건은 계약 문구에 직접 반영한다.
    3. 계약 효력/이행에 중요한데 아직 결정되지 않은 조건은
       반드시 [입력 필요: 조건명] Slot으로 남긴다.
    4. 미정 조건을 애매한 표현으로 숨기지 않는다.

    성능 전략:
    - 계약정보를 필요한 필드만 전달
    - JSON pretty print 제거
    - 법령 본문 길이 제한
    - 불필요한 Prompt 설명 축소
    """

    PROVIDER_NAME = "openai"

    DEFAULT_MODEL = "gpt-6-luna"

    MAX_REFERENCE_CONTENT_CHARS = 1800

    CONTRACT_FIELDS = (
        "principalAmount",
        "interestRate",
        "lateInterestRate",
        "startDate",
        "maturityDate",
        "repaymentDay",
        "repaymentType",
        "contractAlias",
        "terms",
    )

    # API/Spring 경계에서 사용하는 이름과
    # Agent 내부 canonical 이름이 달라도 생성 프롬프트에서
    # 확정 계약값이 누락되지 않도록 alias를 허용한다.
    CONTRACT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "principalAmount": (
            "principalAmount",
            "principal",
        ),
        "interestRate": (
            "interestRate",
        ),
        "lateInterestRate": (
            "lateInterestRate",
        ),
        "startDate": (
            "startDate",
        ),
        "maturityDate": (
            "maturityDate",
            "repaymentDate",
        ),
        "repaymentDay": (
            "repaymentDay",
        ),
        "repaymentType": (
            "repaymentType",
        ),
        "contractAlias": (
            "contractAlias",
        ),
        "terms": (
            "terms",
        ),
    }

    ISSUE_DECISIONS: dict[
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
            "연체 횟수 산정 기준",
            "일부 지급 시 연체 처리 기준",
        ],

        "ACCELERATION": [
            "기한의 이익 상실 또는 일시변제 발생 요건",
            "연속 연체 판단 기준",
            "일부 지급 시 연체 처리 기준",
            "사전 통지 여부",
            "사전 통지 방법",
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
            "연체이자 계산 종료일",
        ],

        "INTEREST": [
            "약정이율",
            "이자 계산 대상",
            "이자 계산 방식",
            "이자 지급일",
        ],

        "REPAYMENT": [
            "상환 방식",
            "상환일",
            "회차별 상환금액",
        ],

        "GUARANTEE": [
            "보증 범위",
            "보증 한도",
            "보증 기간",
        ],

        "COLLATERAL": [
            "담보 대상",
            "담보 범위",
            "담보 제공 시점",
        ],

        "OTHER": [],
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = (
            model
            or os.getenv(
                "OPENAI_LLM_MODEL"
            )
            or self.DEFAULT_MODEL
        )

        resolved_api_key = (
            api_key
            or os.getenv(
                "OPENAI_API_KEY"
            )
        )

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되어 있지 않습니다."
            )

        self.client = OpenAI(
            api_key=resolved_api_key,
        )

        print(
            f"[ClauseGenerator] model={self.model}"
        )

    @property
    def name(
        self,
    ) -> str:
        return self.PROVIDER_NAME

    def generate(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        issue_types = [
            str(
                issue.get(
                    "type",
                    "OTHER",
                )
            ).upper()
            for issue in issues
        ]

        decision_candidates = (
            self._collect_decision_candidates(
                issue_types
            )
        )

        compact_contract = (
            self._compact_contract(
                contract
            )
        )

        compact_issues = (
            self._compact_issues(
                issues
            )
        )

        compact_references = (
            self._compact_references(
                references
            )
        )

        system_prompt = (
            self._build_system_prompt()
        )

        user_prompt = (
            self._build_user_prompt(
                message=message,
                contract=(
                    compact_contract
                ),
                issues=(
                    compact_issues
                ),
                references=(
                    compact_references
                ),
                decision_candidates=(
                    decision_candidates
                ),
            )
        )

        input_chars = (
            len(system_prompt)
            + len(user_prompt)
        )

        print(
            "[PERF][ClauseGenerator] "
            f"input_chars={input_chars} "
            f"references="
            f"{len(compact_references)}"
        )

        started_at = (
            perf_counter()
        )

        response = (
            self.client.responses.create(
                model=self.model,
                instructions=(
                    system_prompt
                ),
                input=(
                    user_prompt
                ),
                **openai_call_options("GENERATOR"),
            )
        )

        api_elapsed = (
            perf_counter()
            - started_at
        )

        raw_text = (
            self._extract_text(
                response
            )
        )

        print(
            "[PERF][ClauseGenerator] "
            f"api="
            f"{api_elapsed * 1000:.0f}ms "
            f"output_chars="
            f"{len(raw_text)}"
        )

        parsed = (
            self._parse_json(
                raw_text
            )
        )

        return (
            self._normalize_result(
                parsed,
                message=message,
            )
        )

    # ==========================================
    # Prompt
    # ==========================================

    def _build_system_prompt(
        self,
    ) -> str:
        return """
개인 간 금전소비대차계약의 추가 조항 초안을 작성한다.

규칙:

1. 사용자 요청과 현재 계약정보에서 확정된 값은 그대로 반영한다.

2. 계약 이행에 필요한 중요한 조건이 정해지지 않았다면
임의로 결정하지 말고 반드시 다음 형식을 사용한다.

[입력 필요: 조건명]

3. 미정 조건을 아래 표현으로 숨기지 않는다.

- 전부 또는 일부
- 필요한 경우
- 상당한 기간
- 적절한 방법
- 별도로 정한다
- 추후 합의한다
- 당사자가 정한다
- 당사자가 합의한다

단, 사용자가 해당 표현 자체를 명시적으로 요청한 경우는 제외한다.

4. 한 Slot에는 하나의 결정값만 넣는다.

5. 검색된 법령 근거에 없는 법률·조문·판례를 만들지 않는다.

6. 법령에 직접 규정되지 않은 내용을
법률상 당연한 효과처럼 단정하지 않는다.

7. 조항에는 AI, 검색, RAG 같은 기술 용어를 쓰지 않는다.

8. 사용자가 요청하지 않은 의무나 제한을 임의로 추가하지 않는다.

반드시 JSON 객체 하나만 출력한다.

{
  "title": "짧은 조항 제목",
  "summary": "조항 목적을 한두 문장으로 설명",
  "suggestedClause": "계약서에 들어갈 실제 조항",
  "warnings": ["필요한 주의사항"],
  "unresolvedDecisions": [
    {
      "name": "Slot 조건명",
      "reason": "사용자가 결정해야 하는 이유"
    }
  ]
}

unresolvedDecisions에는 suggestedClause에 실제로 존재하는
[입력 필요: ...] 항목만 포함한다.

미결정 조건이 없으면 빈 배열을 반환한다.
""".strip()

    def _build_user_prompt(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[
            dict[str, str]
        ],
        references: list[
            dict[str, str]
        ],
        decision_candidates: list[str],
    ) -> str:
        contract_json = (
            json.dumps(
                contract,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        issues_json = (
            json.dumps(
                issues,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        decisions_json = (
            json.dumps(
                decision_candidates,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        references_json = (
            json.dumps(
                references,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        )

        return f"""
사용자 요청:
{message}

계약정보:
{contract_json}

법적 쟁점:
{issues_json}

검토 가능한 미결정 조건 후보:
{decisions_json}

법령 근거:
{references_json}

작성 순서:
1. 사용자 요청에 이미 정해진 조건인지 확인한다.
2. 계약정보에 이미 정해진 조건인지 확인한다.
3. 조항 이행에 꼭 필요한데 미정이면 [입력 필요: ...]로 남긴다.
4. 불필요한 조건은 추가하지 않는다.

위 정보만 사용해 조항을 작성한다.
""".strip()

    # ==========================================
    # Compact Input
    # ==========================================

    def _compact_contract(
        self,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generator에 필요한 계약정보만 canonical field명으로 전달한다.

        API는 현재 principal / repaymentDate를 사용하고,
        Generator prompt는 principalAmount / maturityDate를 사용한다.

        두 계층의 이름 차이 때문에 원금·만기일이 prompt에서
        사라지지 않도록 alias를 이 경계에서 흡수한다.
        """

        result: dict[
            str,
            Any,
        ] = {}

        for field in (
            self.CONTRACT_FIELDS
        ):
            aliases = (
                self.CONTRACT_FIELD_ALIASES.get(
                    field,
                    (field,),
                )
            )

            value: Any = None

            for alias in aliases:
                candidate = (
                    contract.get(
                        alias
                    )
                )

                if candidate is None:
                    continue

                if isinstance(
                    candidate,
                    str,
                ):
                    candidate = (
                        candidate.strip()
                    )

                    if not candidate:
                        continue

                value = candidate
                break

            if value is None:
                continue

            result[
                field
            ] = value

        return result

    @staticmethod
    def _compact_issues(
        issues: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, str]
    ]:
        result: list[
            dict[str, str]
        ] = []

        for issue in issues:
            issue_type = str(
                issue.get(
                    "type",
                    "OTHER",
                )
            ).upper()

            reason = str(
                issue.get(
                    "reason",
                    "",
                )
            ).strip()

            item: dict[
                str,
                str,
            ] = {
                "type": (
                    issue_type
                ),
            }

            if reason:
                item[
                    "reason"
                ] = reason

            result.append(
                item
            )

        return result

    def _compact_references(
        self,
        references: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, str]
    ]:
        result: list[
            dict[str, str]
        ] = []

        for reference in references:
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

            law_name = str(
                reference.get(
                    "lawName"
                )
                or reference.get(
                    "law_name"
                )
                or metadata.get(
                    "law_name"
                )
                or ""
            ).strip()

            article_number = str(
                reference.get(
                    "articleNumber"
                )
                or reference.get(
                    "article_number"
                )
                or metadata.get(
                    "article_number"
                )
                or ""
            ).strip()

            article_title = str(
                reference.get(
                    "articleTitle"
                )
                or reference.get(
                    "article_title"
                )
                or metadata.get(
                    "article_title"
                )
                or ""
            ).strip()

            content = str(
                reference.get(
                    "content"
                )
                or reference.get(
                    "text"
                )
                or ""
            ).strip()

            if (
                len(content)
                > self.MAX_REFERENCE_CONTENT_CHARS
            ):
                content = (
                    content[
                        :self.MAX_REFERENCE_CONTENT_CHARS
                    ].rstrip()
                    + "…"
                )

            item = {
                "law": (
                    law_name
                ),
                "article": (
                    article_number
                ),
                "content": (
                    content
                ),
            }

            if article_title:
                item[
                    "title"
                ] = article_title

            result.append(
                item
            )

        return result

    # ==========================================
    # Decision Candidates
    # ==========================================

    def _collect_decision_candidates(
        self,
        issue_types: list[str],
    ) -> list[str]:
        result: list[str] = []

        for issue_type in (
            issue_types
        ):
            candidates = (
                self.ISSUE_DECISIONS.get(
                    issue_type,
                    [],
                )
            )

            for candidate in (
                candidates
            ):
                if (
                    candidate
                    not in result
                ):
                    result.append(
                        candidate
                    )

        return result

    # ==========================================
    # Response
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
            parts: list[str] = []

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
                        parts.append(
                            str(text)
                        )

            if parts:
                return "\n".join(
                    parts
                ).strip()

        except Exception:
            pass

        raise ValueError(
            "ClauseGenerator 응답에서 "
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
            value = (
                json.loads(
                    cleaned
                )
            )

        except (
            json.JSONDecodeError
        ) as exc:
            raise ValueError(
                "ClauseGenerator가 JSON 형식이 "
                "아닌 응답을 반환했습니다."
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "ClauseGenerator 응답은 "
                "JSON 객체여야 합니다."
            )

        return value

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
        *,
        message: str,
    ) -> dict[str, Any]:
        title = str(
            result.get(
                "title"
            )
            or "추가 약정"
        ).strip()

        summary = str(
            result.get(
                "summary"
            )
            or ""
        ).strip()

        suggested_clause = str(
            result.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        warnings = (
            result.get(
                "warnings"
            )
        )

        if not isinstance(
            warnings,
            list,
        ):
            warnings = []

        normalized_warnings = [
            str(
                item
            ).strip()
            for item in warnings
            if str(
                item
            ).strip()
        ]

        unresolved = (
            result.get(
                "unresolvedDecisions"
            )
        )

        if not isinstance(
            unresolved,
            list,
        ):
            unresolved = []

        normalized_unresolved: list[
            dict[str, str]
        ] = []

        for item in unresolved:
            if not isinstance(
                item,
                dict,
            ):
                continue

            name = str(
                item.get(
                    "name"
                )
                or ""
            ).strip()

            reason = str(
                item.get(
                    "reason"
                )
                or ""
            ).strip()

            if not name:
                continue

            normalized_unresolved.append(
                {
                    "name": name,
                    "reason": reason,
                }
            )

        return {
            "message": (
                message
            ),
            "title": (
                title
            ),
            "summary": (
                summary
            ),
            "suggestedClause": (
                suggested_clause
            ),
            "warnings": (
                normalized_warnings
            ),
            "unresolvedDecisions": (
                normalized_unresolved
            ),
        }