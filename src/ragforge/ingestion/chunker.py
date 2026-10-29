from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200


# A numbered heading on its own line: "4. SECURITY DEPOSIT.", "13. UTILITIES"
# (no trailing period), "12. FIRE SAFETY and HAZARDOUS MATERIALS." (lowercase
# connecting word). _is_heading_title() then requires the title to be mostly
# upper case, so numbered sentences such as "3. Tenant shall pay" are not
# mistaken for headings.
_SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^(?P<number>\d{1,3})\.[ \t]+"
    r"(?P<title>[A-Z][A-Za-z0-9&'()/,\- ]{2,}?)\.?[ \t]*$"
)

_MIN_UPPERCASE_SHARE = 0.7


def _is_heading_title(title: str) -> bool:
    letters = [ch for ch in title if ch.isalpha()]
    if len(letters) < 3:
        return False
    return sum(ch.isupper() for ch in letters) / len(letters) >= _MIN_UPPERCASE_SHARE


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

    matches = [
        match
        for match in _SECTION_HEADING_PATTERN.finditer(text)
        if _is_heading_title(match.group("title"))
    ]

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


def _split_into_structural_units(
    text: str,
) -> list[tuple[str, str | None, int]]:
    """
    Split text into structural units: (text, section_number, start_offset).

    Example:

        ("4. SECURITY DEPOSIT....", "4", 1830)
        ("some preamble...", None, 0)

    start_offset is the position of the unit's first character in ``text``.
    """

    if not text.strip():
        return []

    boundaries = _find_section_boundaries(text)
    spans = [(0, boundaries[0][0] if boundaries else len(text), None)]
    spans += boundaries

    units: list[tuple[str, str | None, int]] = []
    for start, end, section_number in spans:
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            units.append((stripped, section_number, start + len(raw) - len(raw.lstrip())))
    return units


def _merge_pages(
    documents: Sequence[Document],
) -> list[tuple[dict, str, list[tuple[int, dict]]]]:
    """
    Join the pages of each source document into one text.

    PDF loaders return one Document per page. Chunking pages separately cut
    every clause that continues onto the next page into two chunks, and the
    continuation had no section heading. Returns, per source document:
    (base metadata, joined text, [(start_offset, page metadata), ...]).
    Documents keep their input order; pages keep their order within a document.
    """

    groups: dict[tuple, tuple[dict, list[str], list[tuple[int, dict]]]] = {}
    for document in documents:
        metadata = dict(document.metadata)
        key = (metadata.get("source"), metadata.get("document_id"))
        text = remove_page_headers(document.page_content)
        if not text:
            continue
        if key not in groups:
            groups[key] = (metadata, [], [])
        _, texts, page_starts = groups[key]
        offset = sum(len(t) + 1 for t in texts)
        texts.append(text)
        page_starts.append((offset, metadata))

    return [
        (base, "\n".join(texts), page_starts)
        for base, texts, page_starts in groups.values()
    ]


_PAGE_KEYS = ("page", "page_number", "page_label")


def _page_metadata_at(offset: int, page_starts: list[tuple[int, dict]]) -> dict:
    """Page metadata of the page that contains ``offset``."""
    starts = [start for start, _ in page_starts]
    index = max(bisect_right(starts, offset) - 1, 0)
    page = page_starts[index][1]
    return {key: page[key] for key in _PAGE_KEYS if key in page}


def _build_chunk(
    base_metadata: dict,
    text: str,
    section_number: str | None,
    chunk_index: int,
    section_chunk_index: int,
    start_page: dict,
    end_page: dict,
) -> Document:
    """
    Build a LangChain Document representing one chunk.

    page / page_number are the page where the chunk starts; page_end_number
    is the page where it ends (equal unless the chunk crosses a page break).
    """

    metadata = {key: value for key, value in base_metadata.items() if key not in _PAGE_KEYS}
    metadata.update(start_page)
    if "page_number" in end_page:
        metadata["page_end_number"] = end_page["page_number"]

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

    1. Remove page headers and join each document's pages, so a clause
       that continues onto the next page stays in one section.
    2. Detect numbered sections.
    3. Keep complete sections together when possible.
    4. Split only sections that exceed chunk_size.
    5. Preserve section and page metadata (start and end page).
    6. Join a short preamble (a title, an effective date) to the first
       section instead of dropping it: it is too small to stand alone,
       but it can hold the only copy of a fact.
    """

    if not documents:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []
    global_chunk_index = 0

    for base_metadata, text, page_starts in _merge_pages(documents):
        preamble_start: int | None = None
        for unit_text, section_number, unit_start in _split_into_structural_units(text):

            # Unsectioned text too small to be a chunk of its own (a title
            # block before the first heading) is carried into the next unit.
            if section_number is None and len(unit_text) < MIN_CHUNK_SIZE:
                preamble_start = unit_start
                continue
            if preamble_start is not None:
                unit_text = text[preamble_start:unit_start + len(unit_text)].strip()
                unit_start = preamble_start
                preamble_start = None

            if len(unit_text) <= chunk_size:
                pieces = [(unit_text, 0)]
            else:
                pieces = []
                cursor = 0
                for split_document in splitter.create_documents([unit_text]):
                    piece = split_document.page_content.strip()
                    if not piece:
                        continue
                    # Search from where the previous piece ended (minus the
                    # overlap), so repeated wording cannot match an earlier
                    # occurrence and give the piece the wrong page.
                    position = unit_text.find(piece, cursor)
                    if position < 0:
                        position = cursor
                    pieces.append((piece, position))
                    cursor = max(position + 1, position + len(piece) - chunk_overlap)

            for section_chunk_index, (piece, position) in enumerate(pieces):
                piece_start = unit_start + position
                chunks.append(
                    _build_chunk(
                        base_metadata=base_metadata,
                        text=piece,
                        section_number=section_number,
                        chunk_index=global_chunk_index,
                        section_chunk_index=section_chunk_index,
                        start_page=_page_metadata_at(piece_start, page_starts),
                        end_page=_page_metadata_at(piece_start + len(piece) - 1, page_starts),
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