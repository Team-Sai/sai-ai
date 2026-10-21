"""
사용자 요청의 금액·퍼센트·날짜 표현을 '값'으로 바꿔 비교한다.

ClauseValidator는 사용자가 직접 지정한 값이 생성 조항에 반영됐는지 확인한다.
예전에는 같은 글자가 있는지만 봐서, 뜻이 같아도 표기가 다르면 충돌로 판단했다.

    "5만원"        ↔ "50,000원"          → 같은 금액
    "5.5%"         ↔ "연 5.5퍼센트"       → 같은 비율
    "2026.12.31"   ↔ "2026년 12월 31일"   → 같은 날짜
    "오늘로부터 1년 뒤" ↔ "2027년 10월 21일" → 기준일 + 기간으로 계산한 날짜

여기서는 양쪽을 숫자/날짜 값으로 바꾼 뒤 비교한다.
판단할 수 없는 표현은 False를 돌려서 기존처럼 LLM 검증으로 넘긴다(안전한 쪽).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

KST = timezone(timedelta(hours=9))

KOREAN_DIGITS = {
    "영": 0,
    "공": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}

SMALL_UNITS = {
    "십": 10,
    "백": 100,
    "천": 1000,
}

BIG_UNITS = {
    "만": 10_000,
    "억": 100_000_000,
    "조": 1_000_000_000_000,
}

AMOUNT_CHARS = set("0123456789,. ") | set(KOREAN_DIGITS) | set(SMALL_UNITS) | set(BIG_UNITS)

PERCENT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|퍼센트|프로)"
)

DATE_PATTERN = re.compile(
    r"(20\d{2})\s*(?:년|[-./])\s*(\d{1,2})\s*(?:월|[-./])\s*(\d{1,2})\s*일?"
)

RELATIVE_PATTERN = re.compile(
    r"(\d+)\s*(년|개월|달|주|일)\s*(?:뒤|후|이후|지난)"
)


def today_kst() -> date:
    return datetime.now(KST).date()


# ======================================
# 금액
# ======================================

def _parse_amount_phrase(phrase: str) -> int | None:
    """
    "5백만", "1억 2천만", "50,000", "오만", "십만" 같은 표현을 정수로 바꾼다.
    올바른 숫자 표현이 아니면 None.
    """

    text = phrase.replace(" ", "").replace(",", "")

    if not text:
        return None

    total = 0
    section = 0
    current: float | None = None
    index = 0
    seen_value = False

    while index < len(text):
        char = text[index]

        if char.isdigit():
            if current is not None:
                return None

            match = re.match(r"\d+(?:\.\d+)?", text[index:])
            current = float(match.group(0))
            index += len(match.group(0))
            seen_value = True
            continue

        if char in KOREAN_DIGITS:
            if current is not None:
                return None

            current = KOREAN_DIGITS[char]
            index += 1
            seen_value = True
            continue

        if char in SMALL_UNITS:
            section += (current if current is not None else 1) * SMALL_UNITS[char]
            current = None
            index += 1
            seen_value = True
            continue

        if char in BIG_UNITS:
            value = section + (current if current is not None else 0)

            if value == 0:
                value = 1

            total += value * BIG_UNITS[char]
            section = 0
            current = None
            index += 1
            seen_value = True
            continue

        return None

    if not seen_value:
        return None

    total += section + (current if current is not None else 0)

    if total <= 0 or total != int(total):
        return None

    return int(total)


def find_amounts(text: str) -> set[int]:
    """
    문장 안의 '...원' 금액을 모두 찾아 정수로 돌려준다.

    '원' 앞의 숫자 표현을 거꾸로 모은 뒤,
    파싱되는 가장 긴 뒷부분만 사용한다.
    (예: "매월 13일 10만원" → "13일10만"은 실패, "10만"은 성공 → 100,000)
    """

    amounts: set[int] = set()

    for match in re.finditer("원", text):
        end = match.start()
        start = end

        while start > 0 and text[start - 1] in AMOUNT_CHARS:
            start -= 1

        phrase = text[start:end].strip()

        if not phrase:
            continue

        # "이원화", "구원" 같은 일반 단어를 금액으로 오해하지 않도록
        # 아라비아 숫자나 단위(십·백·천·만·억·조)가 있는 표현만 금액으로 본다.
        if not any(
            char.isdigit() or char in SMALL_UNITS or char in BIG_UNITS
            for char in phrase
        ):
            continue

        parts = phrase.split()

        for cut in range(len(parts)):
            value = _parse_amount_phrase("".join(parts[cut:]))

            if value is not None:
                amounts.add(value)
                break

    return amounts


# ======================================
# 퍼센트
# ======================================

def find_percents(text: str) -> set[float]:
    return {
        round(float(value), 4)
        for value in PERCENT_PATTERN.findall(text)
    }


# ======================================
# 날짜
# ======================================

def find_dates(text: str) -> set[date]:
    found: set[date] = set()

    for year, month, day in DATE_PATTERN.findall(text):
        try:
            found.add(date(int(year), int(month), int(day)))
        except ValueError:
            continue

    return found


def _add_period(base: date, amount: int, unit: str) -> date:
    if unit == "일":
        return base + timedelta(days=amount)

    if unit == "주":
        return base + timedelta(weeks=amount)

    months = amount * 12 if unit == "년" else amount
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])

    return date(year, month, day)


def _parse_contract_date(value: Any) -> date | None:
    if not value:
        return None

    dates = find_dates(str(value))

    if dates:
        return next(iter(dates))

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def relative_date_candidates(
    message: str,
    *,
    amount: int,
    unit: str,
    contract: dict[str, Any] | None,
    today: date,
) -> set[date]:
    """
    "1년 뒤"를 실제 날짜 후보로 바꾼다.

    기준일 후보:
    - 오늘 (메시지에 '오늘'이 있거나 기준이 불분명할 때)
    - 계약 시작일 (contract.startDate)
    - 메시지에 적힌 다른 날짜 (예: "지급일 2026년 12월 31일로부터 1년 뒤")
    하루 차이(초일 산입 여부)는 같은 것으로 본다.
    """

    bases: set[date] = {today}

    start = _parse_contract_date((contract or {}).get("startDate"))

    if start:
        bases.add(start)

    bases |= find_dates(message)

    candidates: set[date] = set()

    for base in bases:
        target = _add_period(base, amount, unit)
        candidates |= {
            target - timedelta(days=1),
            target,
            target + timedelta(days=1),
        }

    return candidates


# ======================================
# 공개 함수
# ======================================

def _is_relative_period_token(token: str, message: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"(\d+)(년|개월|달|주|일)", token)

    if not match:
        return None

    for relative in RELATIVE_PATTERN.finditer(message):
        if relative.group(1) == match.group(1) and relative.group(2) == match.group(2):
            return int(match.group(1)), match.group(2)

    return None


def is_value_reflected(
    *,
    token: str,
    message: str,
    clause: str,
    contract: dict[str, Any] | None = None,
    today: date | None = None,
) -> bool:
    """
    사용자가 지정한 값(token)이 조항에 '같은 값'으로 들어 있는지 확인한다.
    값으로 판단할 수 없으면 False (기존 문자열 비교 결과를 따른다).
    """

    today = today or today_kst()

    if token.endswith("원"):
        wanted = find_amounts(token)
        return bool(wanted) and wanted <= find_amounts(clause)

    if token.endswith("%"):
        wanted = find_percents(token)
        return bool(wanted) and wanted <= find_percents(clause)

    token_dates = find_dates(token)

    if token_dates:
        return token_dates <= find_dates(clause)

    relative = _is_relative_period_token(token, message)

    if relative:
        amount, unit = relative
        candidates = relative_date_candidates(
            message,
            amount=amount,
            unit=unit,
            contract=contract,
            today=today,
        )
        return bool(candidates & find_dates(clause))

    return False


def extra_value_tokens(message: str) -> Iterable[str]:
    """
    기존 정규식이 놓치는 금액·퍼센트 표현을 추가로 돌려준다.
    (예: "5백만원", "1억 2천만원", "오만 원", "5.5퍼센트")
    """

    for amount in sorted(find_amounts(message)):
        yield f"{amount}원"

    for match in PERCENT_PATTERN.finditer(message):
        if "%" not in match.group(0):
            yield f"{match.group(1)}%"