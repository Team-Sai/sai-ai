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