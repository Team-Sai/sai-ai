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

    def analyze(
        self,
        contract: dict[str, Any],
        message: str,
    ) -> list[dict[str, Any]]:
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
      "type": "ACCELERATION",
      "reason": "왜 이 쟁점으로 판단했는지 짧은 설명",
      "searchTerms": [
        "잔여 채무 일시 변제",
        "분할상환 연체",
        "기한의 이익"
      ]
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

        # 계약 특약과 무관한 요청은 쟁점을 만들지 않고 빈 목록을 돌려준다.
        # (ContractClauseAgent가 빈 목록을 보고 범위 밖 안내로 바로 종료한다.)
        if parsed.get("inScope") is False:
            return []

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
            return []

        return issues