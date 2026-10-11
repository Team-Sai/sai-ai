from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class ClauseValidator:
    """
    생성된 계약 조항의 최종 안전 검증을 담당한다.

    주요 검증:
    - 사용자 의도 반영 여부
    - 검색 법령 근거와의 일치
    - 현재 계약과의 충돌
    - 모호한 계약조건
    - 미정 계약조건 Slot 처리
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
        Generator 결과를 검토하고,
        필요하면 suggestedClause를 수정한다.
        """

        original_clause = str(
            generated.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        ambiguous_matches = (
            self._find_ambiguous_phrases(
                original_clause
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

        decision_candidates = (
            self._collect_decision_candidates(
                issue_types
            )
        )

        existing_slots = (
            self._extract_slots(
                original_clause
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
                ambiguous_matches=ambiguous_matches,
                decision_candidates=decision_candidates,
                existing_slots=existing_slots,
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

        validated = (
            self._parse_json(
                raw_text
            )
        )

        return self._normalize_result(
            validated=validated,
            generated=generated,
            ambiguous_matches=ambiguous_matches,
        )

    def _build_system_prompt(
        self,
    ) -> str:
        return """
당신은 금전소비대차계약 조항의 최종 검증기다.

Generator가 만든 조항을 그대로 믿지 말고
다음 네 가지를 검증한다.

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


[미결정 계약조건 처리 규칙]

사용자 요청 또는 현재 계약정보에 값이 없는 중요한 계약조건을
임의로 결정해서는 안 된다.

반드시 다음 형식으로 남긴다.

[입력 필요: 조건명]


특히 아래 표현은 주의한다.

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


단, 사용자가 해당 범위나 표현을 명시적으로 요청한 경우에는
그 표현 자체가 확정된 조건일 수 있다.


예:

사용자:
"만기 전에 돈을 먼저 갚을 수 있게 해줘."

잘못된 결과:
"채무자는 원금의 전부 또는 일부를 중도상환할 수 있다."

개선:
"채무자는 최종 만기일 전에
[입력 필요: 중도상환 가능 범위]를 중도상환할 수 있다."


사용자:
"전액이든 일부든 자유롭게 먼저 갚을 수 있게 해줘."

이 경우:
"원금의 전부 또는 일부를 중도상환할 수 있다."

라고 작성해도 된다.


[Slot 규칙]

반드시 정확히:

[입력 필요: 조건명]

형식을 사용한다.

한 Slot 안에는 한 가지 결정값만 넣는다.


[중요]

검증 과정에서 필요 이상의 Slot을 만들지 않는다.

계약의 실제 이행 또는 권리·의무 범위를 결정하는 데
실제로 필요한 조건만 Slot으로 만든다.

법적 근거가 충분하지 않은 경우
조항을 거짓으로 확정하지 말고 warnings에 남긴다.

법령에 존재하지 않는 조문을 만들지 않는다.


[출력]

반드시 JSON 객체 하나만 출력한다.

{
  "suggestedClause": "검증 후 최종 조항",
  "warnings": [
    "최종 사용자에게 보여줄 주의사항"
  ],
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
        ambiguous_matches: list[dict[str, str]],
        decision_candidates: list[str],
        existing_slots: list[str],
    ) -> str:
        reference_summaries: list[
            dict[str, Any]
        ] = []

        for reference in references:
            reference_summaries.append(
                {
                    "lawName":
                        reference.get(
                            "lawName"
                        )
                        or reference.get(
                            "law_name"
                        ),

                    "articleNumber":
                        reference.get(
                            "articleNumber"
                        )
                        or reference.get(
                            "article_number"
                        ),

                    "articleTitle":
                        reference.get(
                            "articleTitle"
                        )
                        or reference.get(
                            "article_title"
                        ),

                    "content":
                        reference.get(
                            "content"
                        )
                        or reference.get(
                            "text"
                        ),
                }
            )

        return f"""
[사용자 요청]
{message}

[현재 계약정보]
{json.dumps(contract, ensure_ascii=False, indent=2)}

[법적 쟁점]
{json.dumps(issues, ensure_ascii=False, indent=2)}

[Generator 결과]
{json.dumps(generated, ensure_ascii=False, indent=2)}

[현재 생성 문장에서 발견된 모호 표현]
{json.dumps(ambiguous_matches, ensure_ascii=False, indent=2)}

[이미 존재하는 입력 Slot]
{json.dumps(existing_slots, ensure_ascii=False, indent=2)}

[해당 쟁점에서 고려할 결정사항 후보]
{json.dumps(decision_candidates, ensure_ascii=False, indent=2)}

후보를 전부 Slot으로 만들 필요는 없다.

다만 실제 계약의 권리·의무나 금액, 시점, 범위를 결정하는 데
필요한 조건인데 사용자와 현재 계약정보 어디에도 값이 없다면
반드시 [입력 필요: ...]로 남긴다.

특히 Generator가
'전부 또는 일부',
'별도로 합의한다',
'당사자가 정한다'
등으로 선택을 회피했다면
사용자가 실제로 그 범위를 요청한 것인지 확인한다.

사용자가 정하지 않았다면
명확한 Slot으로 재작성한다.

[검색 법령]
{json.dumps(reference_summaries, ensure_ascii=False, indent=2)}

위 정보만 이용해 최종 조항을 검증하라.
""".strip()

    def _find_ambiguous_phrases(
        self,
        clause: str,
    ) -> list[dict[str, str]]:
        found: list[
            dict[str, str]
        ] = []

        for (
            phrase,
            reason,
        ) in self.AMBIGUOUS_PATTERNS:
            if phrase in clause:
                found.append(
                    {
                        "phrase":
                            phrase,

                        "reason":
                            reason,
                    }
                )

        return found

    def _collect_decision_candidates(
        self,
        issue_types: list[str],
    ) -> list[str]:
        result: list[str] = []

        for issue_type in issue_types:
            values = (
                self.ISSUE_REQUIRED_DECISIONS.get(
                    issue_type,
                    [],
                )
            )

            for value in values:
                if value not in result:
                    result.append(
                        value
                    )

        return result

    def _extract_slots(
        self,
        clause: str,
    ) -> list[str]:
        return [
            item.strip()
            for item in self.SLOT_PATTERN.findall(
                clause
            )
            if item.strip()
        ]

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

            for item in response.output:
                for content in item.content:
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
            "ClauseValidator 응답에서 텍스트를 찾지 못했습니다."
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
            parsed = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "ClauseValidator가 JSON 형식이 아닌 응답을 반환했습니다."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                "ClauseValidator 응답은 JSON 객체여야 합니다."
            )

        return parsed

    def _normalize_result(
        self,
        *,
        validated: dict[str, Any],
        generated: dict[str, Any],
        ambiguous_matches: list[dict[str, str]],
    ) -> dict[str, Any]:
        suggested_clause = str(
            validated.get(
                "suggestedClause"
            )
            or generated.get(
                "suggestedClause"
            )
            or ""
        ).strip()

        validator_warnings = (
            validated.get(
                "warnings"
            )
        )

        if not isinstance(
            validator_warnings,
            list,
        ):
            validator_warnings = []

        generator_warnings = (
            generated.get(
                "warnings"
            )
        )

        if not isinstance(
            generator_warnings,
            list,
        ):
            generator_warnings = []

        warnings: list[str] = []

        for item in (
            validator_warnings
            + generator_warnings
        ):
            text = str(
                item
            ).strip()

            if (
                text
                and text not in warnings
            ):
                warnings.append(
                    text
                )

        validation = validated.get(
            "validation"
        )

        if not isinstance(
            validation,
            dict,
        ):
            validation = {
                "status":
                    "REVIEW",

                "isSafeToSuggest":
                    True,

                "checks":
                    [],
            }

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
            status = "REVIEW"

        is_safe = bool(
            validation.get(
                "isSafeToSuggest",
                True,
            )
        )

        checks = validation.get(
            "checks"
        )

        if not isinstance(
            checks,
            list,
        ):
            checks = []

        normalized_checks: list[
            dict[str, str]
        ] = []

        allowed_types = {
            "GROUNDING",
            "USER_INTENT",
            "CONTRACT_CONSISTENCY",
            "AMBIGUITY",
        }

        for check in checks:
            if not isinstance(
                check,
                dict,
            ):
                continue

            check_type = str(
                check.get(
                    "type"
                )
                or ""
            ).upper()

            if check_type not in allowed_types:
                continue

            check_status = str(
                check.get(
                    "status"
                )
                or "REVIEW"
            ).upper()

            if check_status not in {
                "PASS",
                "REVIEW",
                "WARNING",
            }:
                check_status = "REVIEW"

            message = str(
                check.get(
                    "message"
                )
                or ""
            ).strip()

            normalized_checks.append(
                {
                    "type":
                        check_type,

                    "status":
                        check_status,

                    "message":
                        message,
                }
            )

        remaining_ambiguity = (
            self._find_ambiguous_phrases(
                suggested_clause
            )
        )

        slot_names = (
            self._extract_slots(
                suggested_clause
            )
        )

        if (
            remaining_ambiguity
            and not slot_names
        ):
            status = "REVIEW"

            ambiguity_message = (
                "계약 문구에 아직 범위나 기준이 "
                "확정되지 않은 표현이 남아 있어 "
                "추가 확인이 필요합니다."
            )

            if (
                ambiguity_message
                not in warnings
            ):
                warnings.append(
                    ambiguity_message
                )

        return {
            "suggestedClause":
                suggested_clause,

            "warnings":
                warnings,

            "validation": {
                "status":
                    status,

                "isSafeToSuggest":
                    is_safe,

                "checks":
                    normalized_checks,
            },

            "debug": {
                "detectedAmbiguousPhrases":
                    ambiguous_matches,

                "remainingAmbiguousPhrases":
                    remaining_ambiguity,

                "slots":
                    slot_names,
            },
        }