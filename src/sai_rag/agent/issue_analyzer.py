import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from sai_rag.agent.contract_context_formatter import (
    ContractContextFormatter,
)

from sai_rag.agent.openai_options import (
    openai_call_options,
)


load_dotenv()


class ClauseIssueAnalyzer:
    """
    사용자의 자연어 특약 요청에서
    법률적으로 검토할 쟁점을 추출한다.
    """

    ALLOWED_ISSUES = {
        "REPAYMENT",
        "DEFAULT",
        "ACCELERATION",
        "INTEREST",
        "LATE_INTEREST",
        "PENALTY",
        "EARLY_REPAYMENT",
        "GUARANTEE",
        "COLLATERAL",
        "EXECUTION",
        "ELECTRONIC_SIGNATURE",
        "OTHER",
    }

    ALLOWED_SLOT_KINDS = {
        "MONTHS",
        "DAYS",
        "COUNT",
        "PERCENT",
        "AMOUNT",
        "DATE",
        "CHOICE",
        "OTHER",
    }

    ALLOWED_SLOT_FIELDS = {
        "principalAmount",
        "interestRate",
        "lateInterestRate",
        "startDate",
        "maturityDate",
        "repaymentDay",
        "repaymentType",
    }


    MAX_QUESTIONS = 3

    def __init__(self):
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되어 있지 않습니다."
            )

        self.model = os.getenv(
            "OPENAI_LLM_MODEL",
            "gpt-6-luna",
        )

        self.client = OpenAI(
            api_key=api_key
        )

    def analyze_detailed(
        self,
        contract: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        if not message.strip():
            raise ValueError(
                "법률 쟁점을 분석할 사용자 요청이 비어 있습니다."
            )

        contract_context = (
            ContractContextFormatter.format(
                contract
            )
        )

        prompt = f"""
당신은 금전소비대차 계약의 법률 쟁점을 분류하는 시스템입니다.

사용자의 요청을 보고 실제 법령 검색에 필요한 법률 쟁점을 추출하세요.

아래 issue type 중에서만 선택할 수 있습니다.

REPAYMENT
- 변제기, 분할상환, 상환 방법, 상환 순서

DEFAULT
- 채무불이행, 이행지체, 연체, 미지급

ACCELERATION
- 잔여 채무의 일시변제, 기한의 이익, 조기 변제 청구

INTEREST
- 약정이자, 최고이자율, 이자 약정

LATE_INTEREST
- 연체이자, 지연손해금, 금전채무불이행 손해배상

PENALTY
- 위약금, 손해배상액의 예정, 과도한 위약금

EARLY_REPAYMENT
- 중도상환, 만기 전 변제

GUARANTEE
- 보증인, 연대보증, 보증채무

COLLATERAL
- 담보, 저당권, 담보 제공

EXECUTION
- 강제집행, 공정증서, 집행

ELECTRONIC_SIGNATURE
- 전자계약, 전자서명, 전자문서

OTHER
- 위 분류로 설명하기 어려운 법률 쟁점


중요한 규칙:

1. 사용자가 말하지 않은 법률 문제를 억지로 추가하지 마세요.

2. 하나의 요청에 여러 쟁점이 있으면 여러 개를 선택할 수 있습니다.

3. 각 쟁점마다 검색에 사용할 핵심 표현을 작성하세요.

4. 특정 법령명이나 조문 번호를 생성하지 마세요.

5. 계약 필드의 의미를 아래 설명대로 정확하게 이해하세요.

6. 최종 만기일과 매월 분할상환일은 서로 다른 개념입니다.

7. 반드시 JSON만 출력하세요.

8. 먼저 사용자 요청이 금전소비대차(돈을 빌려주고 받는) 계약서의
   특약·조항 작성 또는 수정과 관련 있는지 판단하세요.
   관련 없으면(예: 요리, 날씨, 여행, 일반 상식, 코딩,
   부동산 매매·임대차·근로계약 등 다른 종류의 계약)
   "inScope": false 로 두고 "issues"는 빈 배열 [] 로 출력하세요.
   이때 억지로 OTHER 쟁점을 만들지 마세요.

9. 관련 있으면 "inScope": true 로 출력하세요.

10. 사용자가 원하는 내용을 "understanding.summary"에 한두 문장으로 요약하세요.
   "~를 원하시는 걸로 이해했어요."처럼 사용자에게 확인하는 말투로 쓰세요.

11. 사용자 요청에 나온 구체적인 값(기간, 횟수, 금액, 비율, 날짜, 선택)을
   "understanding.slots"에 넣으세요. 요청에 없는 값을 지어내지 마세요.
   kind는 MONTHS, DAYS, COUNT, PERCENT, AMOUNT, DATE, CHOICE, OTHER 중 하나입니다. 어느 것에도 맞지 않으면 OTHER를 쓰세요.
   value에는 사용자가 쓴 단위까지 함께 쓰세요(예: 3개월, 7%, 50만원).
   계약 정보의 값을 바꾸려는 요청이면 slot에 아래 칸도 함께 쓰세요.
   - "value": 바꿀 값
   - "from": 사용자가 말한 원래 값. 말하지 않았으면 쓰지 마세요.
   - "field": 바꾸려는 계약 항목. principalAmount, interestRate, lateInterestRate,
     startDate, maturityDate, repaymentDay, repaymentType 중 하나만 쓰세요.
   - "condition": 값이 바뀌는 조건(예: "2개월 연속 연체 시"). 조건이 없으면 쓰지 마세요.
   계약 항목을 바꾸지 않는 값(연체 개월 수, 보증 한도 등)에는 "field"를 쓰지 마세요.

12. 특약을 쓰기 전에 사용자가 정해야 할 것이 있으면 "questions"에 최대 3개까지 넣으세요.
   - "현재 계약 정보"에 이미 있는 값(원금, 이자율, 시작일, 만기일, 상환일, 상환방식)은 묻지 마세요.
   - 계약 정보에 없는 것(보증인, 담보물, 일부만 낸 경우의 처리 등)만 물으세요.
   - id는 영어 소문자와 밑줄로 짧게 쓰세요.
   - 고를 수 있는 답이 있으면 "options"에 2~4개 넣고, 없으면 []로 두세요.
   - 정할 것이 없으면 "questions"는 []로 두세요.
   "inScope"가 false면 "understanding"과 "questions"는 쓰지 마세요.


현재 계약 정보:

{contract_context}


원본 데이터:

{json.dumps(contract, ensure_ascii=False, indent=2)}


사용자 요청:

{message}


출력 형식:

{{
"inScope": true,
  "issues": [
    {{
      "type": "GUARANTEE",
      "reason": "보증인을 세우려는 요청입니다.",
      "searchTerms": [
        "보증채무",
        "연대보증"
      ]
    }},
    {{
      "type": "COLLATERAL",
      "reason": "물건을 담보로 잡으려는 요청입니다.",
      "searchTerms": [
        "담보 제공",
        "질권",
        "담보물 처분"
      ]
    }}
  ],
  "understanding": {{
    "summary": "친구분을 연대보증인으로 세우고, 노트북을 담보로 잡는 조항을 원하시는 걸로 이해했어요.",
    "slots": [
      {{"name": "보증 방식", "kind": "CHOICE", "value": "연대보증"}},
      {{"name": "담보물", "kind": "CHOICE", "value": "노트북"}}
    ]
  }},
  "questions": [
    {{
      "id": "collateral_handling",
      "text": "제때 갚지 못하면 담보물을 어떻게 할까요?",
      "options": ["팔아서 갚는 데 쓰기", "채권자가 가져가기"]
    }}
  ]
}}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            **openai_call_options("ANALYZER"),
        )

        output = response.output_text.strip()

        try:
            parsed = json.loads(
                output
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "법률 쟁점 분석 결과를 JSON으로 해석하지 못했습니다."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ValueError(
                    "법률 쟁점 분석 결과 형식이 올바르지 않습니다."
            )

        # 계약 특약과 무관한 요청은 쟁점을 만들지 않고 빈 쟁점 목록을 돌려준다.
        # (ContractClauseAgent가 빈 목록을 보고 범위 밖 안내로 바로 종료한다.)
        if parsed.get("inScope") is False:
            return self._out_of_scope_result()

        raw_issues = parsed.get(
            "issues",
            [],
        )

        if not isinstance(
            raw_issues,
            list,
        ):
            raw_issues = []

        issues: list[
            dict[str, Any]
        ] = []

        for raw_issue in raw_issues:
            if not isinstance(
                raw_issue,
                dict,
            ):
                continue

            issue_type = str(
                raw_issue.get(
                    "type",
                    "",
                )
            ).upper()

            if (
                issue_type
                not in self.ALLOWED_ISSUES
            ):
                continue

            reason = str(
                raw_issue.get(
                    "reason",
                    "",
                )
            )

            raw_terms = raw_issue.get(
                "searchTerms",
                [],
            )

            search_terms = [
                str(term).strip()
                for term in raw_terms
                if str(term).strip()
            ]

            issues.append(
                {
                    "type": issue_type,
                    "reason": reason,
                    "searchTerms": (
                        search_terms
                    ),
                }
            )

        # 예전에는 쟁점이 없으면 OTHER를 강제로 넣어 끝까지 진행했지만,
        # 그러면 관련 없는 질문에도 특약이 생성된다.
        # 이제는 빈 목록을 돌려 범위 밖으로 처리한다.
        if not issues:
            return self._out_of_scope_result()

        return {
            "issues": issues,
            "understanding": self._parse_understanding(
                parsed.get("understanding")
            ),
            "questions": self._parse_questions(
                parsed.get("questions")
            ),
        }

    def analyze(
        self,
        contract: dict[str, Any],
        message: str,
    ) -> list[dict[str, Any]]:
        """기존 호출용. 쟁점 목록만 돌려준다."""

        return self.analyze_detailed(
            contract=contract,
            message=message,
        )["issues"]

    @staticmethod
    def _out_of_scope_result() -> dict[str, Any]:
        return {
            "issues": [],
            "understanding": None,
            "questions": [],
        }

    @classmethod
    def _parse_understanding(
        cls,
        raw: Any,
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raw = {}

        raw_slots = raw.get("slots")

        if not isinstance(raw_slots, list):
            raw_slots = []

        slots: list[dict[str, Any]] = []

        for raw_slot in raw_slots:
            if not isinstance(raw_slot, dict):
                continue

            name = str(raw_slot.get("name") or "").strip()
            value = str(raw_slot.get("value") or "").strip()
            kind = str(raw_slot.get("kind") or "").strip().upper()

            if not name or not value:
                continue

            if kind not in cls.ALLOWED_SLOT_KINDS:
                kind = "OTHER"

            from_value = str(raw_slot.get("from") or "").strip()
            field = str(raw_slot.get("field") or "").strip()
            condition = str(raw_slot.get("condition") or "").strip()

            if field not in cls.ALLOWED_SLOT_FIELDS:
                field = ""

            slots.append(
                {
                    "name": name,
                    "kind": kind,
                    "value": value,
                    "from": from_value or None,
                    "field": field or None,
                    "condition": condition or None,
                    # 아래 두 칸은 LLM이 보내도 믿지 않는다.
                    # inMessage는 규칙(slot_rules)이, confirmed는 사용자 답(분류기)이 정한다.
                    "inMessage": False,
                    "confirmed": False,
                }
            )

        return {
            "summary": str(raw.get("summary") or "").strip(),
            "slots": slots,
        }

    @classmethod
    def _parse_questions(
        cls,
        raw: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []

        questions: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for raw_question in raw:
            if not isinstance(raw_question, dict):
                continue

            question_id = str(raw_question.get("id", "")).strip()
            text = str(raw_question.get("text", "")).strip()

            if not question_id or not text or question_id in seen_ids:
                continue

            raw_options = raw_question.get("options")

            if not isinstance(raw_options, list):
                raw_options = []

            options = [
                str(option).strip()
                for option in raw_options
                if str(option).strip()
            ]

            seen_ids.add(question_id)

            questions.append(
                {
                    "id": question_id,
                    "text": text,
                    "options": options,
                }
            )

            if len(questions) >= cls.MAX_QUESTIONS:
                break

        return questions