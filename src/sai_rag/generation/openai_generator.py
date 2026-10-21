import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.sai_rag.generation.base import (
    ReviewGenerator,
)


load_dotenv()


class OpenAIReviewGenerator(
    ReviewGenerator
):

    def __init__(self):
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 없습니다."
            )

        self.model = os.getenv(
            "OPENAI_LLM_MODEL",
            "gpt-6-luna",
        )

        self.client = OpenAI(
            api_key=api_key
        )

        print(
            "[OpenAIReviewGenerator] "
            f"model={self.model}"
        )

    @property
    def name(self) -> str:
        return "openai"

    def generate_review(
        self,
        contract: dict[str, Any],
        rule_results: list[
            dict[str, Any]
        ],
        legal_reviews: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        review_context = (
            self._build_review_context(
                rule_results=
                    rule_results,
                legal_reviews=
                    legal_reviews,
            )
        )

        prompt = self._build_prompt(
            contract=contract,
            review_context=
                review_context,
        )

        response = (
            self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
        )

        generated = (
            self._parse_json(
                response.output_text
            )
        )

        return (
            self._build_final_result(
                generated=generated,
                review_context=
                    review_context,
            )
        )

    @staticmethod
    def _build_review_context(
        rule_results: list[
            dict[str, Any]
        ],
        legal_reviews: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:

        rule_by_field = {
            result["field"]: result
            for result in rule_results
        }

        contexts = []

        for legal_review in legal_reviews:

            field = (
                legal_review["field"]
            )

            rule_result = (
                rule_by_field.get(
                    field
                )
            )

            status = (
                rule_result["status"]
                if rule_result
                else "REVIEW"
            )

            contexts.append(
                {
                    "field":
                        field,

                    "type":
                        legal_review[
                            "type"
                        ],

                    "status":
                        status,

                    "ruleResult":
                        rule_result,

                    "references":
                        legal_review[
                            "references"
                        ],
                }
            )

        return contexts

    @staticmethod
    def _build_prompt(
        contract: dict[str, Any],
        review_context: list[
            dict[str, Any]
        ],
    ) -> str:

        contract_json = json.dumps(
            contract,
            ensure_ascii=False,
            indent=2,
        )

        context_json = json.dumps(
            review_context,
            ensure_ascii=False,
            indent=2,
        )

        return f"""
당신은 금전소비대차 계약의 자동 법률 검토 결과를
사용자가 이해하기 쉽게 설명하는 시스템입니다.

법률 판정자가 아니라 설명 생성기입니다.

반드시 아래 Rule Engine 결과와
검색된 법률 근거만 사용하세요.

규칙:
1. 제공되지 않은 법률이나 조문을 만들지 마세요.
2. 검색 결과에 없는 조문 번호를 인용하지 마세요.
3. PASS, WARNING, ERROR, REVIEW 상태를 변경하지 마세요.
4. REVIEW는 적법하다는 뜻이 아닙니다.
5. REVIEW 상태를 PASS처럼 표현하지 마세요.
6. Rule Engine이 판단하지 않은 내용을
   "문제가 없다", "적법하다"고 표현하지 마세요.
7. 법률 근거만으로 확정하기 어려우면
   추가 확인이 필요하다고 설명하세요.
8. references는 생성하지 마세요.
9. 짧고 명확한 사용자용 문장으로 작성하세요.
10. 반드시 JSON만 출력하세요.

상태 의미:
PASS = 실제 규칙으로 검사하여 기준 통과
WARNING = 수정 또는 확인 필요
ERROR = 입력값 오류
REVIEW = 관련 법률 근거를 찾았지만 추가 검토 필요

계약 입력:
{contract_json}

검토 데이터:
{context_json}

다음 형식만 출력하세요.

{{
  "description": "전체 결과에 대한 짧은 안내",
  "items": [
    {{
      "field": "원래 field",
      "title": "카드 제목",
      "summary": "한 줄 결과",
      "detail": "검색 법률을 바탕으로 한 설명",
      "suggestion": "조치가 필요하면 안내, 아니면 null"
    }}
  ]
}}
""".strip()

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any]:

        cleaned = text.strip()

        if cleaned.startswith(
            "```"
        ):
            cleaned = (
                cleaned
                .replace(
                    "```json",
                    "",
                    1,
                )
                .replace(
                    "```",
                    "",
                )
                .strip()
            )

        try:
            return json.loads(
                cleaned
            )

        except json.JSONDecodeError as e:
            raise ValueError(
                "LLM 응답 JSON 변환 실패\n"
                f"{cleaned}"
            ) from e

    @staticmethod
    def _build_final_result(
        generated: dict[str, Any],
        review_context: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any]:

        generated_by_field = {
            item.get("field"): item
            for item
            in generated.get(
                "items",
                [],
            )
        }

        final_items = []

        for context in review_context:

            field = context[
                "field"
            ]

            generated_item = (
                generated_by_field.get(
                    field,
                    {},
                )
            )

            references = []

            for reference in context[
                "references"
            ]:

                references.append(
                    {
                        "documentId":
                            reference[
                                "documentId"
                            ],

                        "lawName":
                            reference[
                                "lawName"
                            ],

                        "articleNumber":
                            reference[
                                "articleNumber"
                            ],

                        "articleTitle":
                            reference[
                                "articleTitle"
                            ],

                        "score":
                            reference[
                                "score"
                            ],
                    }
                )

            final_items.append(
                {
                    "field":
                        field,

                    "type":
                        context["type"],

                    "status":
                        context[
                            "status"
                        ],

                    "title":
                        generated_item.get(
                            "title"
                        ),

                    "summary":
                        generated_item.get(
                            "summary"
                        ),

                    "detail":
                        generated_item.get(
                            "detail"
                        ),

                    "suggestion":
                        generated_item.get(
                            "suggestion"
                        ),

                    "references":
                        references,
                }
            )

        passed = sum(
            item["status"] == "PASS"
            for item in final_items
        )

        warning = sum(
            item["status"]
            == "WARNING"
            for item in final_items
        )

        error = sum(
            item["status"] == "ERROR"
            for item in final_items
        )

        review = sum(
            item["status"]
            == "REVIEW"
            for item in final_items
        )

        total = len(
            final_items
        )

        parts = []

        if passed:
            parts.append(
                f"{passed}개 적합"
            )

        if warning:
            parts.append(
                f"{warning}개 수정 필요"
            )

        if error:
            parts.append(
                f"{error}개 입력 오류"
            )

        if review:
            parts.append(
                f"{review}개 확인 필요"
            )

        summary_message = (
            f"{total}개 항목 중 "
            + " · ".join(parts)
        )

        return {
            "summary": {
                "total":
                    total,

                "passed":
                    passed,

                "warning":
                    warning,

                "error":
                    error,

                "review":
                    review,

                "message":
                    summary_message,

                "description":
                    generated.get(
                        "description"
                    ),
            },

            "items":
                final_items,

            "disclaimer": (
                "본 결과는 입력된 정보와 "
                "수집된 법령을 기반으로 한 "
                "자동 검토 결과입니다."
            ),
        }