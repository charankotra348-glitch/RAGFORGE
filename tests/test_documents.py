"""Document upload, listing and deletion."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException

from ragforge.app import documents
from ragforge.app.cache.answers import get_corpus_version
from ragforge.app.main import app
from ragforge.core.config import get_settings

from fakes import FakeRedis

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = (ROOT / "data" / "test.pdf").read_bytes()
TEXT = ("1. RENT.\nTenant shall pay the monthly rent of $900 on the first day of each month.\n" * 5).encode()


@pytest.fixture()
def env(monkeypatch, tmp_path, api_headers):
    redis = FakeRedis()
    monkeypatch.setattr(app.state, "redis", redis, raising=False)
    monkeypatch.setattr(app.state, "corpus_version", None, raising=False)
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    calls = []

    def fake_reindex(path, settings):
        calls.append(Path(path))
        return SimpleNamespace(document_id="abc123", source=str(path), chunks=7, pages=2, corpus_version="5")

    monkeypatch.setattr(documents, "reindex", fake_reindex)
    return SimpleNamespace(client=TestClient(app), headers=api_headers, redis=redis, calls=calls, upload_dir=tmp_path / "uploads")


def upload(env, name, data, headers=None):
    return env.client.post("/documents", files={"file": (name, data)}, headers=env.headers if headers is None else headers)


def test_upload_pdf_indexes_it_and_stores_the_file(env) -> None:
    response = upload(env, "My Lease (final).pdf", SAMPLE_PDF)
    assert response.status_code == 201
    body = response.json()
    assert body == {"document_id": "abc123", "filename": "My_Lease_final_.pdf", "chunks": 7, "pages": 2, "replaced_previous_version": False}
    assert env.calls == [env.upload_dir.resolve() / "My_Lease_final_.pdf"]
    assert (env.upload_dir / "My_Lease_final_.pdf").read_bytes() == SAMPLE_PDF
    assert app.state.corpus_version == "5"
    assert sorted(p.name for p in env.upload_dir.iterdir()) == ["My_Lease_final_.pdf"]  # no temp files left


def test_same_name_replaces_previous_version(env) -> None:
    upload(env, "notes.txt", TEXT)
    response = upload(env, "notes.txt", TEXT.replace(b"$900", b"$950"))
    assert response.json()["replaced_previous_version"] is True
    assert b"$950" in (env.upload_dir / "notes.txt").read_bytes()


def test_path_traversal_name_stays_inside_upload_dir(env) -> None:
    response = upload(env, "../../etc/passwd.txt", TEXT)
    assert response.status_code == 201
    assert response.json()["filename"] == "passwd.txt"
    assert (env.upload_dir / "passwd.txt").exists()


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("script.sh", b"echo hi", 415),
        ("fake.pdf", b"just text", 415),
        ("empty.txt", b"   ", 422),
        ("binary.txt", b"\xff\xfe\x00bad", 415),
        ("broken.pdf", b"%PDF-1.4 not really a pdf", 422),
        (".pdf", SAMPLE_PDF, 415),
    ],
)
def test_bad_files_are_rejected_without_indexing(env, name, data, status) -> None:
    response = upload(env, name, data)
    assert response.status_code == status
    assert env.calls == []
    assert not env.upload_dir.exists() or list(env.upload_dir.iterdir()) == []


def test_too_large_file_is_rejected(env, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    assert upload(env, "notes.txt", TEXT).status_code == 413
    assert env.calls == []


def test_failed_indexing_restores_previous_version(env, monkeypatch) -> None:
    upload(env, "notes.txt", TEXT)

    def broken(path, settings):
        raise ResponseHandlingException(ConnectionError("qdrant down"))

    monkeypatch.setattr(documents, "reindex", broken)
    response = upload(env, "notes.txt", TEXT.replace(b"$900", b"$950"))
    assert response.status_code == 503
    assert b"$900" in (env.upload_dir / "notes.txt").read_bytes()
    assert sorted(p.name for p in env.upload_dir.iterdir()) == ["notes.txt"]


def test_failed_indexing_of_new_file_leaves_nothing(env, monkeypatch) -> None:
    monkeypatch.setattr(documents, "reindex", lambda path, settings: (_ for _ in ()).throw(ConnectionError("down")))
    assert upload(env, "new.txt", TEXT).status_code == 503
    assert list(env.upload_dir.iterdir()) == []


def test_upload_requires_api_key(env) -> None:
    assert upload(env, "notes.txt", TEXT, headers={}).status_code == 401
    assert env.calls == []


def indexed(*items):
    return [
        Document(page_content="x", metadata={"document_id": doc_id, "source": source, "page_number": page})
        for doc_id, source, page in items
    ]


def test_list_documents_groups_chunks(env, monkeypatch) -> None:
    monkeypatch.setattr(documents, "get_qdrant", lambda settings: object())
    monkeypatch.setattr(
        documents,
        "get_indexed_chunks",
        lambda client, settings: indexed(("b1", "/d/lease.pdf", 1), ("b1", "/d/lease.pdf", 4), ("a1", "/u/notes.txt", 1)),
    )
    response = env.client.get("/documents", headers=env.headers)
    assert response.json() == [
        {"document_id": "b1", "filename": "lease.pdf", "chunks": 2, "pages": 4},
        {"document_id": "a1", "filename": "notes.txt", "chunks": 1, "pages": 1},
    ]


def test_list_documents_returns_503_when_qdrant_is_down(env, monkeypatch) -> None:
    def down(settings):
        raise ResponseHandlingException(ConnectionError("refused to secret-host"))

    monkeypatch.setattr(documents, "get_qdrant", down)
    response = env.client.get("/documents", headers=env.headers)
    assert response.status_code == 503 and "secret-host" not in response.text


def test_delete_document(env, monkeypatch) -> None:
    upload(env, "notes.txt", TEXT)
    stored = env.upload_dir.resolve() / "notes.txt"
    outside = env.upload_dir.parent / "keep.pdf"
    outside.write_bytes(b"%PDF-")
    deleted = []
    monkeypatch.setattr(documents, "get_qdrant", lambda settings: object())
    monkeypatch.setattr(documents, "count_document_chunks", lambda client, settings, doc_id: 3 if doc_id == "abc123" else 0)
    monkeypatch.setattr(
        documents, "get_indexed_chunks", lambda client, settings: indexed(("abc123", str(stored), 1), ("abc123", str(outside), 1))
    )
    monkeypatch.setattr(documents, "delete_document", lambda client, settings, doc_id: deleted.append(doc_id))
    before = get_corpus_version(env.redis)

    assert env.client.delete("/documents/not a valid id!", headers=env.headers).status_code in (404, 422)
    assert env.client.delete("/documents/unknown", headers=env.headers).status_code == 404
    response = env.client.delete("/documents/abc123", headers=env.headers)

    assert response.json() == {"document_id": "abc123", "deleted_chunks": 3}
    assert deleted == ["abc123"]
    assert not stored.exists() and outside.exists()  # only files inside UPLOAD_DIR are removed
    assert int(get_corpus_version(env.redis)) == int(before) + 1


def test_safe_filename() -> None:
    assert documents.safe_filename("Lease 2026 (v2).PDF") == "Lease_2026_v2_.pdf"
    with pytest.raises(Exception):
        documents.safe_filename("archive.zip")
