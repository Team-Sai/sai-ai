"""
생성 모델이 조각으로 보내는 JSON에서 조항 글자만 골라내는지 확인한다.
"""

import json

import pytest

from sai_rag.agent.clause_delta import ClauseTextExtractor


def feed_all(chunks):
    extractor = ClauseTextExtractor()
    return "".join(extractor.feed(chunk) for chunk in chunks), extractor


def test_reads_only_suggested_clause_value():
    text, extractor = feed_all(['{"suggestedClause": "제1조', " 채무자는", ' 변제한다.", "title": "x"}'])

    assert text == "제1조 채무자는 변제한다."
    assert extractor.finished


def test_key_split_across_chunks():
    text, _ = feed_all(['{"sugges', 'tedClau', 'se"', ' : "', "안녕", '"}'])

    assert text == "안녕"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_escapes_split_anywhere(size):
    clause = '제1조\n"상환"은 \\ 역슬래시, 탭\t, 한글 \u00e9'
    raw = json.dumps({"title": "t", "suggestedClause": clause}, ensure_ascii=True)
    chunks = [raw[i:i + size] for i in range(0, len(raw), size)]

    text, extractor = feed_all(chunks)

    assert text == clause
    assert extractor.finished


def test_ignores_text_after_clause_ends():
    extractor = ClauseTextExtractor()

    assert extractor.feed('{"suggestedClause": "끝"') == "끝"
    assert extractor.feed(', "summary": "다른 글"}') == ""



# ---------------------------------------------------------------
# 리뷰 반영: 서로게이트 쌍(이모지)과 잘못된 이스케이프
# ---------------------------------------------------------------

@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11])
def test_surrogate_pair_split_anywhere(size):
    clause = "상환 완료 😀 축하"   # 😀 = "\ud83d\ude00"
    raw = json.dumps({"suggestedClause": clause}, ensure_ascii=True)
    chunks = [raw[i:i + size] for i in range(0, len(raw), size)]

    text, extractor = feed_all(chunks)

    assert text == clause
    assert extractor.finished


def test_lone_surrogates_are_dropped():
    # 짝이 없는 앞쪽(\ud83d)·뒤쪽(\ude00) 덩어리는 깨진 글자가 되므로 버린다
    text, _ = feed_all(['{"suggestedClause": "가\\ud83d나\\ude00다"}'])

    assert text == "가나다"


def test_invalid_unicode_escape_does_not_swallow_closing_quote():
    # "\uZZ" 뒤의 따옴표까지 삼키면 조항 끝을 놓친다
    extractor = ClauseTextExtractor()

    assert extractor.feed('{"suggestedClause": "끝\\uZZ"') == "끝ZZ"
    assert extractor.finished
    assert extractor.feed(', "summary": "다른 글"}') == ""


def test_unknown_escape_keeps_character():
    text, _ = feed_all(['{"suggestedClause": "a\\xb"}'])

    assert text == "axb"

