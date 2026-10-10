"""Document IDs come from file contents, and re-indexing replaces older copies."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from ragforge.app.retrieval.vector_store import replace_document_chunks
from ragforge.ingestion.loader import _default_document_id


def test_same_file_gets_same_id_from_relative_and_absolute_paths(tmp_path, monkeypatch) -> None:
    file = tmp_path / "lease.txt"
    file.write_text("Monthly rent is $750.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _default_document_id("lease.txt") == _default_document_id(str(file.resolve()))


def test_moved_file_keeps_its_id_and_edited_file_gets_a_new_one(tmp_path) -> None:
    a = tmp_path / "a.txt"
    a.write_text("Monthly rent is $750.", encoding="utf-8")
    original = _default_document_id(str(a))
    moved = tmp_path / "sub" / "a.txt"
    moved.parent.mkdir()
    os.replace(a, moved)
    assert _default_document_id(str(moved)) == original
    moved.write_text("Monthly rent is $800.", encoding="utf-8")
    assert _default_document_id(str(moved)) != original


def test_non_file_source_falls_back_to_hashing_the_string() -> None:
    assert _default_document_id("https://example.com/a") == _default_document_id("https://example.com/a")
    assert _default_document_id("https://example.com/a") != _default_document_id("https://example.com/b")


def _chunks(document_id: str, source: str, n: int = 2) -> list[Document]:
    return [
        Document(page_content=f"chunk {i}", metadata={"document_id": document_id, "source": source, "chunk_index": i})
        for i in range(n)
    ]


def test_replace_removes_older_copy_of_the_same_source() -> None:
    client = MagicMock()
    client.scroll.return_value = ([], None)
    client.count.return_value = SimpleNamespace(count=27)
    settings = SimpleNamespace(qdrant_collection="test", vector_size=1)

    replace_document_chunks(client, settings, _chunks("newid", "/repo/data/test.pdf"), [[0.0], [0.0]])

    client.upsert.assert_called_once()
    count_filter = client.count.call_args.kwargs["count_filter"]
    assert count_filter.must[0].key == "metadata.source"
    assert count_filter.must[0].match.value == "/repo/data/test.pdf"
    assert count_filter.must_not[0].match.value == "newid"
    deleted = client.delete.call_args.kwargs["points_selector"]
    assert deleted is count_filter


def test_replace_deletes_nothing_extra_when_there_is_no_older_copy() -> None:
    client = MagicMock()
    client.scroll.return_value = ([], None)
    client.count.return_value = SimpleNamespace(count=0)
    settings = SimpleNamespace(qdrant_collection="test", vector_size=1)

    replace_document_chunks(client, settings, _chunks("newid", "/repo/data/test.pdf"), [[0.0], [0.0]])

    client.delete.assert_not_called()
