"""
사용자 의도 검증의 금액·퍼센트·날짜 정규화 테스트.

같은 값을 다른 글자로 쓴 경우는 충돌이 아니어야 하고,
값이 실제로 다르면 여전히 충돌이어야 한다(안전장치 유지).
"""

import os
from datetime import date

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from sai_rag.agent import intent_value_normalizer as normalizer  # noqa: E402
from sai_rag.agent.clause_validator import ClauseValidator  # noqa: E402

TODAY = date(2026, 10, 21)
CONTRACT = {"startDate": "2026-11-01"}


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.setattr(normalizer, "today_kst", lambda: TODAY)
    return ClauseValidator(api_key="test-key-not-real")


def conflicts(validator, message, clause):
    return validator._find_user_intent_conflicts(
        message=message,
        clause=clause,
        contract=CONTRACT,
    )


# ---------------------------------------------------------------
# 1. 값 추출
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("5만원", {50_000}),
        ("50,000원", {50_000}),
        ("5만 원", {50_000}),
        ("오만 원", {50_000}),
        ("십만원", {100_000}),
        ("5백만원", {5_000_000}),
        ("금 오백만 원", {5_000_000}),
        ("1억 2천만원", {120_000_000}),
        ("3천만 원", {30_000_000}),
        ("매월 13일 10만원", {100_000}),
        ("원금 1,000,000원", {1_000_000}),
        ("이원화", set()),
    ],
)
def test_find_amounts(text, expected):
    assert normalizer.find_amounts(text) == expected


def test_find_percents():
    assert normalizer.find_percents("연 5.5% 또는 5.5퍼센트, 10프로") == {5.5, 10.0}


def test_find_dates_various_formats():
    assert normalizer.find_dates("2026.12.31 / 2026년 12월 31일 / 2026. 12. 31.") == {date(2026, 12, 31)}


# ---------------------------------------------------------------
# 2. 같은 값, 다른 표기 → 충돌 아님
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "message, clause",
    [
        ("약속을 어기면 위약금 10만원을 내도록 해줘", "채무자는 위약금 100,000원을 지급한다."),
        ("중도상환하면 수수료로 5만원만 받게 해줘", "중도상환수수료로 금 50,000원을 지급한다."),
        ("5백만원을 빌려주는 조항", "대여금 5,000,000원을 대여한다."),
        ("지연이자는 연 12%로 해줘", "연 12퍼센트의 지연손해금을 지급한다."),
        ("지급일은 2026년 12월 31일로 해줘", "지급일은 2026. 12. 31.로 한다."),
        ("최종 변제기일은 오늘로부터 1년 뒤로 해줘", "차주는 2027년 10월 21일까지 변제한다."),
        ("최종 변제기일은 계약일로부터 6개월 후로 해줘", "차주는 2027년 5월 1일까지 변제한다."),
        ("지급일 2026년 12월 31일로부터 1년 뒤에 갚게 해줘", "대여금은 2026년 12월 31일에 지급하고, 차주는 2027년 12월 31일까지 변제한다."),
        ("매월 13일에 갚게 해줘", "매월 13일에 상환한다."),
    ],
)
def test_same_value_different_notation_is_not_conflict(validator, message, clause):
    assert conflicts(validator, message, clause) == []


# ---------------------------------------------------------------
# 3. 값이 다르면 여전히 충돌 (안전장치 유지)
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "message, clause",
    [
        ("중도상환하면 수수료로 5만원만 받게 해줘", "중도상환수수료로 금 60,000원을 지급한다."),
        ("약속을 어기면 위약금 10만원을 내도록 해줘", "채무자는 위약금 1,000,000원을 지급한다."),
        ("지연이자는 연 12%로 해줘", "연 15%의 지연손해금을 지급한다."),
        ("지급일은 2026년 12월 31일로 해줘", "지급일은 2026년 11월 30일로 한다."),
        ("최종 변제기일은 오늘로부터 1년 뒤로 해줘", "차주는 2028년 3월 1일까지 변제한다."),
        ("매월 13일에 갚게 해줘", "매월 15일에 상환한다."),
    ],
)
def test_different_value_is_still_conflict(validator, message, clause):
    assert conflicts(validator, message, clause)


def test_amount_written_only_in_korean_is_checked(validator):
    # 기존 정규식은 "5백만원"을 못 뽑아 검사 자체를 건너뛰었다 → 이제 검사한다
    assert conflicts(validator, "5백만원을 빌려주는 조항", "대여금 3,000,000원을 대여한다.")


# ---------------------------------------------------------------
# 4. 생성 프롬프트 규칙
# ---------------------------------------------------------------

def test_generator_prompt_has_acceleration_and_format_rules():
    from sai_rag.agent.clause_generator import ClauseGenerator

    prompt = ClauseGenerator(api_key="test-key-not-real")._build_system_prompt()
    assert "즉시 변제" in prompt
    assert "50,000원" in prompt
    assert "실제 날짜로 계산" in prompt


# ---------------------------------------------------------------
# 5. 잘못된 숫자 표현은 금액으로 보지 않는다 (리뷰 반영)
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["이삼원", "오오만원", "백천원", "천천원", "만억원", "5만5만원"],
)
def test_invalid_korean_number_is_not_amount(text):
    assert normalizer.find_amounts(text) == set()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("일억 이천만 원", {120_000_000}),
        ("삼천오백만원", {35_000_000}),
        ("1조 2억원", {1_000_200_000_000}),
        ("원금과 지원금은 없다", set()),
    ],
)
def test_valid_unit_order_still_parses(text, expected):
    assert normalizer.find_amounts(text) == expected
