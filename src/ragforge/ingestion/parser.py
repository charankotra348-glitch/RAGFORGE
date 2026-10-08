import re
from typing import List
from langchain_core.documents import Document


def strip_running_headers_footers(text: str) -> str:
    """
    Remove repetitive headers/footers, standalone page numbers,
    and boilerplate page noise commonly produced by PyPDF.
    """
    # Remove lines that are just single page numbers (e.g., "12" or "Page 12 of 45")
    text = re.sub(r"(?m)^\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", "", text, flags=re.IGNORECASE)

    # Remove common confidential / boilerplate footers
    text = re.sub(r"(?m)^\s*(?:CONFIDENTIAL|ALL RIGHTS RESERVED|DRAFT)\s*$", "", text, flags=re.IGNORECASE)

    return text


def repair_hyphenated_words(text: str) -> str:
    """Rejoins words split across line breaks with hyphens (e.g., 'agree-\nment' -> 'agreement')."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def normalize_line_breaks(text: str) -> str:
    """
    Converts mid-paragraph soft line breaks into spaces while preserving
    double line breaks that denote actual paragraphs.
    """
    # Replace single newlines that are not preceded or followed by double newlines with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple consecutive blank lines down to double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_document(doc: Document) -> Document:
    """
    Parse and extract structural content from a single raw Document object.
    """
    content = doc.page_content

    content = strip_running_headers_footers(content)
    content = repair_hyphenated_words(content)
    content = normalize_line_breaks(content)

    # Return a new Document instance with structured text and preserved metadata
    return Document(
        page_content=content,
        metadata=dict(doc.metadata)
    )


def parse_documents(documents: List[Document]) -> List[Document]:
    """Parse a list of loaded documents sequentially."""
    return [parse_document(doc) for doc in documents if doc.page_content.strip()]