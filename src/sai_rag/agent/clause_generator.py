from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ClauseGenerator:
    """
    계약 조항 초안을 생성한다.

    핵심 원칙:
    1. 검색된 법령 근거 안에서만 법적 설명을 작성한다.
    2. 사용자가 이미 정한 조건은 계약 문구에 직접 반영한다.
    3. 계약 효력/이행에 중요한데 아직 결정되지 않은 조건은
       반드시 [입력 필요: 조건명] 형식의 Slot으로 남긴다.
    4. 미정 조건을 '전부 또는 일부', '별도 합의',
       '당사자가 정한다' 같은 표현으로 얼버무리지 않는다.
    """

    PROVIDER_NAME = "openai"

    DEFAULT_MODEL = "gpt-6-luna"

    ISSUE_DECISIONS: dict[str, list[str]] = {
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

        print(
            f"[ClauseGenerator] model={self.model}"
        )

    @property
    def name(self) -> str:
        return self.PROVIDER_NAME

    def generate(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        사용자 요청 + 계약정보 + 쟁점 + 검색 법령으로
        하나의 계약 조항 초안을 생성한다.
        """

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

        system_prompt = (
            self._build_system_prompt()
        )

        user_prompt = (
            self._build_user_prompt(
                message=message,
                contract=contract,
                issues=issues,
                references=references,
                decision_candidates=decision_candidates,
            )
        )

        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
        )

        raw_text = (
            self._extract_text(
                response
            )
        )

        parsed = (
            self._parse_json(
                raw_text
            )
        )

        return self._normalize_result(
            parsed,
            message=message,
        )

    def _build_system_prompt(
        self,
    ) -> str:
        return """
당신은 개인 간 금전소비대차계약의 조항 초안을 작성하는 계약 조항 작성기다.

역할은 법률상 최종 판단을 내리는 것이 아니라,
사용자 요청과 현재 계약정보를 바탕으로 검색된 법령 근거 안에서
계약서에 넣을 수 있는 조항 초안을 만드는 것이다.

반드시 아래 규칙을 지켜라.

[가장 중요한 원칙]

1. 사용자가 명시적으로 정한 값이나 조건은 그대로 반영한다.

2. 현재 계약정보나 사용자 요청으로 결정되지 않은 계약조건을
   임의로 만들어서는 안 된다.

3. 계약의 실제 이행, 금액 계산, 의무 발생 시점,
   당사자의 권리·의무 범위를 결정하는 중요한 조건이 미정이면
   반드시 아래 형식으로 문장 안에 남긴다.

   [입력 필요: 조건명]

4. 아래와 같은 표현으로 미정 조건을 숨기면 안 된다.

   - 전부 또는 일부
   - 필요한 경우
   - 상당한 기간
   - 적절한 방법
   - 별도로 정한다
   - 추후 합의한다
   - 당사자가 합의한다
   - 당사자가 정한다
   - 별도 서면 합의에 따른다

   단, 사용자가 실제로 위 표현 자체를 요청한 경우는 제외한다.

5. 예를 들어 사용자가 단순히
   "만기 전에 원금을 먼저 갚을 수 있게 해줘"
   라고 요청했다고 해서

   "원금의 전부 또는 일부를 변제할 수 있다."

   로 임의 확정하면 안 된다.

   중도상환 범위가 결정되지 않았다면

   "채무자는 최종 만기일 전에
   [입력 필요: 중도상환 가능 범위]를 중도상환할 수 있다."

   처럼 작성한다.

6. 사용자가 이미
   "전액이든 일부든 자유롭게 갚을 수 있게"
   라고 요청했다면
   그 부분은 확정된 조건이므로 Slot으로 만들지 않는다.

7. 모든 Slot은 반드시 정확히 다음 문법을 사용한다.

   [입력 필요: 조건명]

   대괄호만 쓰거나
   [조건명]
   형태로 만들지 않는다.

8. 하나의 Slot 안에는 하나의 판단값만 넣는다.

   나쁜 예:
   [입력 필요: 통지 방법과 유예기간]

   좋은 예:
   [입력 필요: 통지 방법]
   [입력 필요: 변제 유예기간]

9. Slot 이름은 짧고 사용자가 이해할 수 있어야 한다.

10. 법령에 직접 규정되어 있지 않은 사항을
    법률상 당연한 효과처럼 단정하지 않는다.

11. 검색 근거에 없는 법률, 조문, 판례를 만들지 않는다.

12. 조항 자체에는 기술 용어인 RAG, 검색, AI 등의 표현을 넣지 않는다.

[출력]

반드시 JSON 객체 하나만 출력한다.

{
  "title": "조항 제목",
  "summary": "이 조항이 왜 필요한지 쉬운 설명",
  "suggestedClause": "실제 계약서 조항 문구",
  "warnings": [
    "사용자가 계약 전에 확인해야 할 내용"
  ],
  "unresolvedDecisions": [
    {
      "name": "Slot 조건명",
      "reason": "왜 사용자가 결정해야 하는지"
    }
  ]
}

unresolvedDecisions에는 suggestedClause 안의
[입력 필요: ...] 항목만 넣는다.

미결정 조건이 없으면 빈 배열로 출력한다.
""".strip()

    def _build_user_prompt(
        self,
        *,
        message: str,
        contract: dict[str, Any],
        issues: list[dict[str, Any]],
        references: list[dict[str, Any]],
        decision_candidates: list[str],
    ) -> str:
        reference_texts: list[str] = []

        for index, reference in enumerate(
            references,
            start=1,
        ):
            law_name = (
                reference.get("lawName")
                or reference.get("law_name")
                or ""
            )

            article_number = (
                reference.get("articleNumber")
                or reference.get(
                    "article_number"
                )
                or ""
            )

            article_title = (
                reference.get("articleTitle")
                or reference.get(
                    "article_title"
                )
                or ""
            )

            content = (
                reference.get("content")
                or reference.get("text")
                or ""
            )

            reference_texts.append(
                "\n".join(
                    [
                        f"[근거 {index}]",
                        f"법령: {law_name}",
                        (
                            "조문: "
                            f"{article_number} "
                            f"{article_title}"
                        ).strip(),
                        f"내용: {content}",
                    ]
                )
            )

        joined_references = (
            "\n\n".join(
                reference_texts
            )
        )

        contract_json = (
            json.dumps(
                contract,
                ensure_ascii=False,
                indent=2,
            )
        )

        issues_json = (
            json.dumps(
                issues,
                ensure_ascii=False,
                indent=2,
            )
        )

        decisions_json = (
            json.dumps(
                decision_candidates,
                ensure_ascii=False,
                indent=2,
            )
        )

        return f"""
[사용자 요청]
{message}

[현재 계약정보]
{contract_json}

[분석된 법적 쟁점]
{issues_json}

[이 쟁점에서 계약 작성 시 검토할 수 있는 결정사항 후보]
{decisions_json}

주의:
위 결정사항 후보를 전부 Slot으로 만들라는 뜻은 아니다.

반드시 아래 순서로 판단한다.

1. 사용자 메시지에 이미 값이 있는가?
2. 현재 계약정보에 이미 값이 있는가?
3. 조항의 실제 이행에 반드시 필요한 결정사항인가?
4. 그렇다면 미정인 항목만 [입력 필요: ...]로 남긴다.
5. 조항 작성에 꼭 필요하지 않은 부수사항은 억지로 추가하지 않는다.

[검색된 법령 근거]
{joined_references}

위 정보만 사용해 조항 초안을 작성하라.
""".strip()

    def _collect_decision_candidates(
        self,
        issue_types: list[str],
    ) -> list[str]:
        result: list[str] = []

        for issue_type in issue_types:
            candidates = (
                self.ISSUE_DECISIONS.get(
                    issue_type,
                    [],
                )
            )

            for candidate in candidates:
                if candidate not in result:
                    result.append(
                        candidate
                    )

        return result

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

            for item in response.output:
                for content in item.content:
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
            "ClauseGenerator 응답에서 텍스트를 찾지 못했습니다."
        )

    @staticmethod
    def _parse_json(
        raw_text: str,
    ) -> dict[str, Any]:
        cleaned = (
            raw_text
            .strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        try:
            value = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "ClauseGenerator가 JSON 형식이 아닌 응답을 반환했습니다."
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "ClauseGenerator 응답은 JSON 객체여야 합니다."
            )

        return value

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
        *,
        message: str,
    ) -> dict[str, Any]:
        title = str(
            result.get("title")
            or "추가 약정"
        ).strip()

        summary = str(
            result.get("summary")
            or ""
        ).strip()

        suggested_clause = str(
            result.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        warnings = result.get(
            "warnings"
        )

        if not isinstance(
            warnings,
            list,
        ):
            warnings = []

        normalized_warnings = [
            str(item).strip()
            for item in warnings
            if str(item).strip()
        ]

        unresolved = result.get(
            "unresolvedDecisions"
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
                item.get("name")
                or ""
            ).strip()

            reason = str(
                item.get("reason")
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
            "message": message,
            "title": title,
            "summary": summary,
            "suggestedClause": suggested_clause,
            "warnings": normalized_warnings,
            "unresolvedDecisions": normalized_unresolved,
        }