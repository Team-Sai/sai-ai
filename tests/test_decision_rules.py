"""
정해야 할 것 분류 표 테스트.

표가 생성기 후보와 어긋나지 않는지, 질문·안내·정책값·조문 확인 표시가 빠지지 않았는지 지킨다.
"""

from sai_rag.agent.clause_generator import ClauseGenerator
from sai_rag.agent.decision_rules import (
    DECISIONS,
    QUESTION,
    decisions_for,
    notices_for,
    policy_values,
    questions_for,
    unverified_legal_refs,
)


def test_every_generator_candidate_has_a_rule():
    # 생성기 후보 이름이 바뀌면 표도 같이 고치도록, 빠진 이름이 있으면 실패한다.
    for issue_type, names in ClauseGenerator.ISSUE_DECISIONS.items():
        covered = {d.name for d in DECISIONS.get(issue_type, ())}
        missing = [name for name in names if name not in covered]

        assert missing == [], f"{issue_type} 분류 표에 없음: {missing}"


def test_questions_have_ids_texts_and_two_or_more_options():
    questions = [d for ds in DECISIONS.values() for d in ds if d.kind == QUESTION]
    ids = [q.question_id for q in questions]

    assert all(q.question_id and q.question and len(q.options) >= 2 for q in questions)
    assert len(set(ids)) == len(set(q.name for q in questions))


def test_shared_question_is_asked_once():
    names = [d.name for d in decisions_for(["DEFAULT", "ACCELERATION"])]

    assert names.count("일부 지급 시 연체 처리 기준") == 1


def test_questions_without_default_come_first():
    questions = questions_for(["PENALTY"])

    assert questions[0].name == "위약금과 지연손해금의 관계"
    assert not questions[0].has_default
    assert questions[-1].has_default


def test_guarantee_always_has_legal_notices():
    refs = [n.legal_ref for n in notices_for(["GUARANTEE"])]

    assert "민법 제428조의2제1항" in refs
    assert "보증인 보호를 위한 특별법 제4조" in refs
    assert notices_for(["DEFAULT"]) == []


def test_policy_values_are_listed():
    texts = [d.text for d in policy_values()]

    assert any("7일" in text for text in texts)
    assert any("연 20%" in text for text in texts)


def test_unverified_legal_refs_are_reported():
    refs = unverified_legal_refs()

    assert any("연체이자 적용 여부 확인 필요" in ref for ref in refs)
    assert all("민법 제398조" not in ref for ref in refs)


def guide(issue_types, message="", questions=(), answers=None):
    from sai_rag.agent.decision_rules import writing_guide

    return writing_guide(
        issue_types=issue_types,
        message=message,
        questions=list(questions),
        answers=answers or {},
        unknown_answer="UNKNOWN",
    )


def test_unasked_question_with_default_becomes_default_text():
    result = guide(["DEFAULT"])

    assert "그 회차 상환금을 전혀 지급하지 않은 경우에만 연체로 본다" in result["defaults"]
    assert result["open"] == []


def test_answered_or_unknown_question_is_handled():
    answered = guide(["DEFAULT"], answers={"partial_payment": "일부만 내도 연체로 봐요"})
    unknown = guide(["DEFAULT"], answers={"partial_payment": "UNKNOWN"})

    assert all("전혀 지급하지" not in text for text in answered["defaults"])
    assert unknown["open"] == ["일부 지급 시 연체 처리 기준"]


def test_answer_to_similar_llm_question_is_used():
    questions = [{"id": "llm_partial", "text": "일부만 낸 달도 연체로 볼까요?"}]

    result = guide(["DEFAULT"], questions=questions, answers={"llm_partial": "예"})

    assert all("전혀 지급하지" not in text for text in result["defaults"])
    assert "일부 지급 시 연체 처리 기준" not in result["open"]


def test_rejected_defaults_become_open_and_contract_values_are_listed():
    result = guide(["LATE_INTEREST"], answers={"confirm_defaults": "바꿀래요"})

    assert result["defaults"] == []
    assert "연체이자 계산 시작일" in result["open"]
    assert result["contract"] == ["연체이자율: 계약의 연체이율 또는 사용자가 말한 연체이율"]

