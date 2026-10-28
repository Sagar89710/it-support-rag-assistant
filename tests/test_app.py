from app import answer_question, load_documents


def test_knowledge_base_sections_are_loaded() -> None:
    sections = {document.metadata["section"] for document in load_documents()}
    assert "Password Reset" in sections
    assert "Incident Escalation" in sections


def test_supported_question_returns_cited_answer() -> None:
    result = answer_question("How do I reset my password?")
    assert result["validation"] == "passed"
    assert any("Password Reset" in source for source in result["sources"])


def test_unsupported_question_uses_fallback() -> None:
    result = answer_question("What is the cafeteria menu?")
    assert result["validation"] == "fallback"
    assert result["sources"] == []
