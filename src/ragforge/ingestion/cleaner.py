from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from langchain_core.documents import Document


# Characters that are invisible/control characters and should not appear
# in normal extracted document text.
_CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

# Three or more consecutive blank lines are treated as excessive.
_EXCESSIVE_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")

# Multiple spaces/tabs within a line.
_MULTIPLE_SPACES = re.compile(r"[ \t]+")


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters into a consistent representation.

    NFKC handles compatibility characters commonly produced by PDFs,
    while preserving normal readable text.
    """
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable control characters while preserving newlines.
    """
    return _CONTROL_CHARACTERS.sub("", text)


def normalize_line_endings(text: str) -> str:
    """
    Normalize Windows/Mac-style line endings to '\\n'.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_line_whitespace(text: str) -> str:
    """
    Normalize spaces and tabs within each line without destroying
    paragraph or heading boundaries.
    """
    lines = text.split("\n")

    normalized_lines = []

    for line in lines:
        line = _MULTIPLE_SPACES.sub(" ", line)
        line = line.strip()
        normalized_lines.append(line)

    return "\n".join(normalized_lines)


def collapse_blank_lines(text: str) -> str:
    """
    Collapse excessive blank lines while preserving paragraph separation.
    """
    text = _EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def remove_line_break_hyphenation(text: str) -> str:
    """
    Rejoin words split by a hyphen at a line boundary.

    Example:
        'reason-\nable'
        -> 'reasonable'

    We only apply this when the hyphen occurs directly at a line break
    between alphabetic characters.
    """
    return re.sub(
        r"(?<=[A-Za-z])-\n(?=[a-z])",
        "",
        text,
    )


def clean_text(text: str) -> str:
    """
    Apply deterministic, meaning-preserving cleaning to extracted text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return ""

    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = normalize_line_endings(text)

    # Fix words broken across lines, e.g.:
    # "agree-\nment" -> "agreement"
    text = remove_line_break_hyphenation(text)

    # Normalize whitespace without merging meaningful lines.
    text = normalize_line_whitespace(text)

    # Remove excessive blank lines.
    text = collapse_blank_lines(text)

    return text


def clean_document(document: Document) -> Document:
    """
    Clean a single LangChain Document.

    Metadata is copied unchanged so source/page/document information
    remains available for citations and traceability.
    """

    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document")

    cleaned_text = clean_text(document.page_content)

    return Document(
        page_content=cleaned_text,
        metadata=dict(document.metadata),
    )


def clean_documents(documents: Sequence[Document]) -> list[Document]:
    """
    Clean multiple documents and remove documents that become empty.
    """

    cleaned_documents: list[Document] = []

    for document in documents:
        cleaned_document = clean_document(document)

        if cleaned_document.page_content:
            cleaned_documents.append(cleaned_document)

    return cleaned_documents


def validate_cleaned_documents(documents: Sequence[Document]) -> None:
    """
    Validate basic invariants of the cleaning stage.

    Raises ValueError if the cleaning pipeline produces invalid output.
    """

    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise ValueError(
                f"Document at index {index} is not a LangChain Document"
            )

        if not document.page_content.strip():
            raise ValueError(
                f"Document at index {index} contains empty page content"
            )

        if not isinstance(document.metadata, dict):
            raise ValueError(
                f"Document at index {index} has invalid metadata"
            )