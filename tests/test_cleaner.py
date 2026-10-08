from langchain_core.documents import Document
import pytest

from ragforge.ingestion.cleaner import (
    clean_document,
    clean_documents,
    clean_text,
    validate_cleaned_documents,
)


def test_normalize_whitespace():
    text = "Tenant     shall\tcomply with the Lease."

    result = clean_text(text)

    assert result == "Tenant shall comply with the Lease."


def test_preserve_paragraph_structure():
    text = """6. USE OF PREMISES.

Compliance & Conduct: Tenant shall comply with all laws.

Smoke-Free Policy: Smoking is prohibited."""

    result = clean_text(text)

    assert "6. USE OF PREMISES." in result
    assert "Compliance & Conduct:" in result
    assert "Smoke-Free Policy:" in result
    assert "\n\n" in result


def test_remove_control_characters():
    text = "Tenant\x00 shall\x01 comply."

    result = clean_text(text)

    assert result == "Tenant shall comply."


def test_normalize_unicode():
    text = "Tenant shall pay \uFB01fty dollars."

    result = clean_text(text)

    assert result == "Tenant shall pay fifty dollars."


def test_fix_hyphenated_line_break():
    text = "The Tenant shall provide noti-\nfication."

    result = clean_text(text)

    assert result == "The Tenant shall provide notification."


def test_preserve_normal_hyphen():
    text = "This is a non-smoking property."

    result = clean_text(text)

    assert result == "This is a non-smoking property."


def test_clean_document_preserves_metadata():
    document = Document(
        page_content="Tenant     shall comply.",
        metadata={
            "document_id": "test-document",
            "page_number": 1,
        },
    )

    result = clean_document(document)

    assert result.page_content == "Tenant shall comply."
    assert result.metadata == {
        "document_id": "test-document",
        "page_number": 1,
    }


def test_empty_documents_are_removed():
    documents = [
        Document(page_content=""),
        Document(page_content="   "),
        Document(page_content="Valid document."),
    ]

    result = clean_documents(documents)

    assert len(result) == 1
    assert result[0].page_content == "Valid document."


def test_validate_cleaned_documents():
    documents = [
        Document(
            page_content="Valid document.",
            metadata={"document_id": "test"},
        )
    ]

    validate_cleaned_documents(documents)


def test_invalid_document_type():
    with pytest.raises(TypeError):
        clean_text(123)


def test_clean_document_invalid_type():
    with pytest.raises(TypeError):
        clean_document("not a document")