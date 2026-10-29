"""Section detection and page joining in the chunker."""

from langchain_core.documents import Document

from ragforge.ingestion.chunker import _find_section_boundaries, chunk_documents, validate_chunks


def page(text: str, number: int, source: str = "lease.pdf", document_id: str = "doc1") -> Document:
    return Document(
        page_content=f"Page {number}/9\n{text}",
        metadata={"source": source, "document_id": document_id, "page": number - 1, "page_number": number},
    )


def sections(text: str) -> list[str]:
    return [number for _, _, number in _find_section_boundaries(text)]


def test_detects_headings_with_lowercase_words_or_no_period() -> None:
    text = (
        "4. SECURITY DEPOSIT.\nbody\n"
        "12. FIRE SAFETY and HAZARDOUS MATERIALS.\nbody\n"
        "13. UTILITIES\nbody\n"
        "22.  SECURITY SYSTEMS AND DISCLAIMER.\nbody\n"
    )
    assert sections(text) == ["4", "12", "13", "22"]


def test_numbered_sentences_are_not_headings() -> None:
    text = "1. PREMISES.\n3. Tenant shall pay rent monthly.\n4. The landlord may enter\n"
    assert sections(text) == ["1"]


def test_clause_continuing_onto_next_page_stays_in_one_chunk() -> None:
    pages = [
        page("1. PREMISES.\nThe premises are one bedroom.\n2. UTILITIES\nTenant pays water,", 1),
        page("electricity, and snow removal.\n3. RENT.\nRent is due monthly.", 2),
    ]
    chunks = chunk_documents(pages)
    utilities = [c for c in chunks if c.metadata["section_number"] == "2"]
    assert len(utilities) == 1
    assert "water,\nelectricity" in utilities[0].page_content
    assert utilities[0].metadata["page_number"] == 1
    assert utilities[0].metadata["page_end_number"] == 2
    rent = next(c for c in chunks if c.metadata["section_number"] == "3")
    assert (rent.metadata["page_number"], rent.metadata["page_end_number"]) == (2, 2)
    assert rent.metadata["page"] == 1


def test_short_continuation_on_next_page_is_kept() -> None:
    # Previously the per-page chunker dropped "of deductions." as a tiny fragment.
    pages = [page("20. DEPOSIT NOT RENT.\nDeductions follow the required itemized notice", 1), page("of deductions.", 2)]
    chunks = chunk_documents(pages)
    assert any("notice\nof deductions." in c.page_content for c in chunks)


def test_short_preamble_is_joined_to_the_first_section() -> None:
    # A title block shorter than a chunk used to be dropped, losing facts such as an
    # effective date that appear nowhere else in the document.
    pages = [page("ACME POLICY\nEffective March 1, 2026.\n\n1. SCOPE\nThis policy applies to staff.\n2. HOURS\nCore hours are 11 to 3.", 1)]
    chunks = chunk_documents(pages)
    first = chunks[0]
    assert first.metadata["section_number"] == "1"
    assert first.page_content.startswith("ACME POLICY\nEffective March 1, 2026.")
    assert "This policy applies to staff." in first.page_content
    assert all("Effective" not in c.page_content for c in chunks[1:])


def test_different_documents_are_not_merged() -> None:
    pages = [
        page("1. PREMISES.\n" + "a" * 300, 1, source="a.pdf", document_id="a"),
        page("1. PREMISES.\n" + "b" * 300, 1, source="b.pdf", document_id="b"),
    ]
    chunks = chunk_documents(pages)
    assert [c.metadata["document_id"] for c in chunks] == ["a", "b"]


def test_long_section_split_pieces_get_their_own_pages() -> None:
    sentence = "Tenant shall keep the premises clean and sanitary at all times. "
    pages = [page("1. MAINTENANCE.\n" + sentence * 12, 1), page(sentence * 12, 2)]
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    validate_chunks(chunks, chunk_size=500)
    assert len(chunks) > 2
    assert all(c.metadata["section_number"] == "1" for c in chunks)
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[-1].metadata["page_end_number"] == 2
    assert [c.metadata["section_chunk_index"] for c in chunks] == list(range(len(chunks)))
