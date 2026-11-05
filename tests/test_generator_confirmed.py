"""
생성 프롬프트에 들어가는 '사용자가 확정한 조건' 글 테스트.
"""

from sai_rag.agent.chat_router import UNKNOWN_ANSWER
from sai_rag.agent.clause_generator import ClauseGenerator


def test_confirmed_slots_and_answers_become_prompt_lines():
    text = ClauseGenerator._format_confirmed(
        {
            "slots": [
                {
                    "name": "연체 시 이자율",
                    "value": "7%",
                    "from": "5%",
                    "condition": "2개월 연속 연체 시",
                    "confirmed": True,
                },
                {"name": "확정 안 된 값", "value": "1%", "confirmed": False},
            ],
            "questions": [
                {"id": "scope", "text": "어디에 적용할까요?"},
                {"id": "guarantee", "text": "보증인을 둘까요?"},
                {"id": "not_answered", "text": "답 안 한 질문"},
            ],
            "answers": {"scope": "남은 원금 전체", "guarantee": UNKNOWN_ANSWER},
        }
    )

    assert "- 연체 시 이자율: 7% (조건: 2개월 연속 연체 시) (원래 값: 5%)" in text
    assert "확정 안 된 값" not in text
    assert "- 어디에 적용할까요? → 남은 원금 전체" in text
    assert "보증인을 둘까요? → 아직 정하지 않음" in text
    assert "[입력 필요" in text
    assert "답 안 한 질문" not in text


def test_no_confirmed_values_means_empty_text():
    assert ClauseGenerator._format_confirmed(None) == ""
    assert ClauseGenerator._format_confirmed({"slots": [], "questions": [], "answers": {}}) == ""

def test_items_that_are_not_dicts_are_skipped():
    text = ClauseGenerator._format_confirmed(
        {
            "slots": [None, "값", {"name": "이자율", "value": "7%", "confirmed": True}],
            "questions": [None, {"id": "scope", "text": "어디에 적용할까요?"}],
            "answers": {"scope": "남은 원금"},
        }
    )

    assert text == "- 이자율: 7%\n- 어디에 적용할까요? → 남은 원금"


def test_writing_guide_text_has_sections_and_no_slot_rule():
    text = ClauseGenerator._format_writing_guide(
        {"contract": ["약정이율: 계약의 약정이율"], "defaults": ["이자는 남은 원금에 대해 계산한다"], "open": []}
    )

    assert "계약 정보에서 그대로 쓸 값" in text
    assert "- 이자는 남은 원금에 대해 계산한다" in text
    assert "[입력 필요] 칸을 하나도 만들지 않는다" in text
    assert ClauseGenerator._format_writing_guide(None) == ""


def test_article_heading_is_removed_from_clause_body():
    result = ClauseGenerator._normalize_result(
        {"suggestedClause": "제○조(형의 연대보증) ① 형은 채무를 연대보증한다.", "title": "연대보증"},
        message="형이 보증인이 되게 해 주세요",
    )

    assert result["suggestedClause"] == "① 형은 채무를 연대보증한다."


def test_clause_without_heading_is_unchanged():
    result = ClauseGenerator._normalize_result(
        {"suggestedClause": "채무자는 매월 20일에 갚는다.", "title": "상환"},
        message="매월 20일에 갚게 해 주세요",
    )

    assert result["suggestedClause"] == "채무자는 매월 20일에 갚는다."


def test_article_heading_with_dot_colon_or_brackets_is_removed():
    for heading in ("제1조. (중도상환) ", "제 2 조: ", "제1조 [중도상환] ", "제○조【중도상환】"):
        result = ClauseGenerator._normalize_result(
            {"suggestedClause": heading + "채무자는 언제든지 상환할 수 있다.", "title": "중도상환"},
            message="중도상환",
        )

        assert result["suggestedClause"] == "채무자는 언제든지 상환할 수 있다."


def test_sentence_starting_with_je_is_not_removed():
    result = ClauseGenerator._normalize_result(
        {"suggestedClause": "제한 조건 없이 언제든지 상환할 수 있다.", "title": "중도상환"},
        message="중도상환",
    )

    assert result["suggestedClause"] == "제한 조건 없이 언제든지 상환할 수 있다."

