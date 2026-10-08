from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200


_SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^(?P<number>\d{1,3})\.\s+"
    r"(?P<title>[A-Z][A-Z0-9&'()/,\- ]{2,})\.\s*$"
)


_PAGE_HEADER_PATTERN = re.compile(
    r"(?mi)^\s*Page\s+\d+\s*/\s*\d+\s*$"
)


def validate_chunking_parameters(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Validate chunking configuration.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size must be at least {MIN_CHUNK_SIZE}"
        )


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create the recursive text splitter used for long sections.
    """

    validate_chunking_parameters(
        chunk_size,
        chunk_overlap,
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


def remove_page_headers(text: str) -> str:
    """
    Remove page-number headers such as:

        Page 2 / 4
    """

    cleaned = _PAGE_HEADER_PATTERN.sub("", text)

    return cleaned.strip()


def _find_section_boundaries(
    text: str,
) -> list[tuple[int, int, str]]:
    """
    Find legal-document section headings such as:

        4. SECURITY DEPOSIT.
        19. SURRENDER OF PREMISES.
        20. DEFAULT.

    Returns:

        [
            (start_position, end_position, section_number),
            ...
        ]
    """

    matches = list(
        _SECTION_HEADING_PATTERN.finditer(text)
    )

    if not matches:
        return []

    sections: list[tuple[int, int, str]] = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section_number = match.group("number")

        sections.append(
            (
                start,
                end,
                section_number,
            )
        )

    return sections


def _split_document_into_structural_units(
    document: Document,
) -> list[tuple[str, str | None]]:
    """
    Split a document into structural units.

    Each unit is:

        (text, section_number)

    Example:

        ("4. SECURITY DEPOSIT....", "4")
        ("5. ENTRY....", "5")
        ("some preamble...", None)
    """

    text = remove_page_headers(
        document.page_content
    )

    if not text:
        return []

    boundaries = _find_section_boundaries(text)

    if not boundaries:
        return [(text, None)]

    units: list[tuple[str, str | None]] = []

    first_start = boundaries[0][0]

    preamble = text[:first_start].strip()

    if preamble:
        units.append(
            (
                preamble,
                None,
            )
        )

    for start, end, section_number in boundaries:
        section_text = text[start:end].strip()

        if section_text:
            units.append(
                (
                    section_text,
                    section_number,
                )
            )

    return units


def _build_chunk(
    document: Document,
    text: str,
    section_number: str | None,
    chunk_index: int,
    section_chunk_index: int,
) -> Document:
    """
    Build a LangChain Document representing one chunk.
    """

    metadata = dict(document.metadata)

    metadata.update(
        {
            "chunk_index": chunk_index,
            "section_number": section_number,
            "section_chunk_index": section_chunk_index,
            "chunk_type": (
                "section"
                if section_number
                else "preamble"
            ),
        }
    )

    return Document(
        page_content=text.strip(),
        metadata=metadata,
    )


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Chunk documents while preserving legal-document structure.

    Strategy:

    1. Remove page headers.
    2. Detect numbered sections.
    3. Keep complete sections together when possible.
    4. Split only sections that exceed chunk_size.
    5. Preserve section metadata.
    6. Ignore tiny unsectioned PDF-extraction fragments.
    """

    if not documents:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []

    global_chunk_index = 0

    for document in documents:
        structural_units = (
            _split_document_into_structural_units(
                document
            )
        )

        for text, section_number in structural_units:

            # -------------------------------------------------
            # Ignore tiny orphan fragments produced by PDF
            # extraction.
            #
            # Example:
            #
            #     "of deductions."
            #
            # This is not a meaningful standalone section.
            # -------------------------------------------------
            if (
                section_number is None
                and len(text.strip()) < MIN_CHUNK_SIZE
            ):
                continue

            # -------------------------------------------------
            # If the entire section fits inside chunk_size,
            # keep it as one chunk.
            # -------------------------------------------------
            if len(text) <= chunk_size:
                chunks.append(
                    _build_chunk(
                        document=document,
                        text=text,
                        section_number=section_number,
                        chunk_index=global_chunk_index,
                        section_chunk_index=0,
                    )
                )

                global_chunk_index += 1

                continue

            # -------------------------------------------------
            # Section is too large.
            #
            # Split it using RecursiveCharacterTextSplitter.
            # -------------------------------------------------
            split_documents = splitter.create_documents(
                [text]
            )

            for (
                section_chunk_index,
                split_document,
            ) in enumerate(split_documents):

                chunk_text = (
                    split_document.page_content.strip()
                )

                if not chunk_text:
                    continue

                chunks.append(
                    _build_chunk(
                        document=document,
                        text=chunk_text,
                        section_number=section_number,
                        chunk_index=global_chunk_index,
                        section_chunk_index=section_chunk_index,
                    )
                )

                global_chunk_index += 1

    return chunks


def validate_chunks(
    chunks: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """
    Validate generated chunks.
    """

    validate_chunking_parameters(
        chunk_size=chunk_size,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    seen_indices: set[int] = set()

    for chunk in chunks:

        if not isinstance(chunk, Document):
            raise TypeError(
                "Every chunk must be a LangChain Document"
            )

        if not chunk.page_content.strip():
            raise ValueError(
                "Chunk cannot contain empty text"
            )

        if len(chunk.page_content) > chunk_size:
            raise ValueError(
                "Chunk exceeds configured chunk_size: "
                f"{len(chunk.page_content)} > {chunk_size}"
            )

        chunk_index = chunk.metadata.get(
            "chunk_index"
        )

        if not isinstance(chunk_index, int):
            raise ValueError(
                "Every chunk must contain an integer "
                "chunk_index"
            )

        if chunk_index in seen_indices:
            raise ValueError(
                f"Duplicate chunk_index detected: "
                f"{chunk_index}"
            )

        seen_indices.add(chunk_index)