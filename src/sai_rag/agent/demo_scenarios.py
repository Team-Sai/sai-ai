"""
발표 시연 모드.

요청에 demo 표시가 있고, 서버에서 SAI_DEMO_MODE=1로 허용했고, 퀵 질문과 같은 문장일 때만
LLM 대신 config/demo_scenarios.json에 정해 둔 내용을 정해 둔 시간에 맞춰 낸다.
질문·기본값·안내(규칙), 답장 분류(규칙), 근거 법령 검색은 진짜로 돈다.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

DEMO_ENV = "SAI_DEMO_MODE"
CONFIG_ENV = "SAI_DEMO_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "demo_scenarios.json"

# 테스트에서 0으로 바꿔 기다리지 않게 한다.
SPEED = 1.0

STREAM_CHUNK_CHARS = 6


def demo_allowed() -> bool:
    return os.getenv(DEMO_ENV, "").strip().lower() in {"1", "true", "on"}


def demo_key(message: str) -> str:
    # 띄어쓰기·끝 문장부호 차이는 같은 질문으로 본다.
    return re.sub(r"[\s.!?~]", "", message or "")


def _load() -> dict[str, dict[str, Any]]:
    path = Path(os.getenv(CONFIG_ENV, "") or DEFAULT_CONFIG)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[시연] 시나리오 파일을 읽지 못했어요: {type(exc).__name__}")
        return {}

    return {demo_key(item["question"]): item for item in data.get("scenarios", [])}


def find_demo_scenario(message: str, demo: bool) -> dict[str, Any] | None:
    """발표 표시 + 서버 허용 + 퀵 질문이면 시나리오(복사본), 아니면 None."""

    if not demo or not demo_allowed():
        return None

    # 매번 파일을 읽어서, 시간을 고치면 서버를 다시 켜지 않아도 바로 반영된다.
    scenario = _load().get(demo_key(message))

    return copy.deepcopy(scenario) if scenario is not None else None


def seconds(scenario: dict[str, Any], name: str) -> float:
    """시나리오의 timing 값(초). 없으면 0."""

    return max(float(scenario.get("timing", {}).get(name, 0)) * SPEED, 0.0)