import os
from typing import Any


def openai_call_options(
    prefix: str,
) -> dict[str, Any]:
    """
    OpenAI Responses API 호출 옵션을 환경변수로 받는다.

    설정하지 않으면 빈 dict를 돌려줘서 기존 호출과 똑같이 동작한다.

    - OPENAI_{prefix}_REASONING_EFFORT   : 추론형 모델의 생각 깊이 (예: low)
    - OPENAI_{prefix}_MAX_OUTPUT_TOKENS  : 최대 출력 토큰 수 (예: 2000)

    reasoning 옵션은 추론형 모델에서만 지원되므로
    사용 중인 모델이 지원하는지 확인한 뒤 설정한다.
    """

    options: dict[str, Any] = {}

    effort = os.getenv(
        f"OPENAI_{prefix}_REASONING_EFFORT",
        "",
    ).strip()

    if effort:
        options["reasoning"] = {
            "effort": effort,
        }

    max_output_tokens = os.getenv(
        f"OPENAI_{prefix}_MAX_OUTPUT_TOKENS",
        "",
    ).strip()

    if max_output_tokens:
        try:
            parsed_tokens = int(
                max_output_tokens
            )
        except ValueError:
            parsed_tokens = None

        if parsed_tokens is not None and parsed_tokens > 0:
            options["max_output_tokens"] = parsed_tokens
        else:
            # 잘못된 값이면 옵션을 붙이지 않고 기본 동작으로 진행한다.
            print(
                "[OpenAIOptions][WARNING] "
                f"OPENAI_{prefix}_MAX_OUTPUT_TOKENS 값이 올바르지 않아 무시합니다: "
                f"{max_output_tokens!r}"
            )

    return options