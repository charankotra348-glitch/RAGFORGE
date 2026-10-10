"""Document management endpoints.

POST   /documents                 upload a PDF, .txt or .md file and index it
GET    /documents                 list indexed documents
DELETE /documents/{document_id}   remove a document from the index

Uploads are saved under UPLOAD_DIR with a sanitized file name. Uploading a
file with the same name again replaces the earlier version: the new chunks
are indexed first, then the old version's chunks are removed (see
replace_document_chunks). Indexing runs inside the request, which is fine
for documents of a few pages (about 2 s here) but would move to a background
job for large files.

Every change bumps the corpus version, so all API workers rebuild their
keyword index and stop serving cached answers for the old content.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ragforge.app.cache.answers import bump_corpus_version
from ragforge.app.generation.service import clear_hybrid_searcher_cache
from ragforge.app.rate_limit import enforce_rate_limit
from ragforge.app.retrieval.vector import get_qdrant
from ragforge.app.retrieval.vector_store import count_document_chunks, delete_document, get_indexed_chunks
from ragforge.core.config import get_settings
from ragforge.ingestion.chunker import chunk_documents
from ragforge.ingestion.loader import SUPPORTED_EXTENSIONS, ingest_file
from ragforge.vector.reindex import reindex

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_READ_BLOCK = 1024 * 1024


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int


class UploadResult(DocumentInfo):
    replaced_previous_version: bool


class DeleteResult(BaseModel):
    document_id: str
    deleted_chunks: int


def safe_filename(name: str | None) -> str:
    """Base name with unsafe characters replaced; rejects hidden or empty names."""
    base = Path(name or "").name
    cleaned = _UNSAFE_CHARS.sub("_", base).strip("._")[:100]
    suffix = Path(base).suffix.lower()
    if not cleaned or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file. Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )
    stem = Path(cleaned).stem or "document"
    return f"{stem}{suffix}"


def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while block := upload.file.read(_READ_BLOCK):
        data.extend(block)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File is larger than {max_bytes // (1024 * 1024)} MB.")
    return bytes(data)


def _check_content(filename: str, data: bytes) -> None:
    if not data.strip():
        raise HTTPException(status_code=422, detail="File is empty.")
    if filename.endswith(".pdf") and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="File has a .pdf name but is not a PDF.")
    if not filename.endswith(".pdf"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=415, detail="Text files must be UTF-8.") from None


def _check_extractable(path: Path, filename: str) -> None:
    """Parse and chunk the file before touching the index, so a bad file is a 422, not a 500.

    The staged copy keeps a hidden name, which the loader may reject by
    extension; it is parsed through a temporary link with the real suffix.
    """
    probe = path.with_name(f".probe-{filename}")
    try:
        os.link(path, probe)
    except OSError:
        probe.write_bytes(path.read_bytes())
    try:
        chunks = chunk_documents(ingest_file(probe))
    except Exception:
        logger.warning("Uploaded file could not be parsed", exc_info=True)
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.") from None
    finally:
        probe.unlink(missing_ok=True)
    if not chunks:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")


def _corpus_changed(request: Request) -> str | None:
    version = bump_corpus_version(getattr(request.app.state, "redis", None))
    clear_hybrid_searcher_cache()
    request.app.state.corpus_version = version
    return version


@router.post("", status_code=201, response_model=UploadResult)
def upload_document(request: Request, file: UploadFile, key_id: str = Depends(enforce_rate_limit)) -> UploadResult:
    settings = get_settings()
    filename = safe_filename(file.filename)
    data = _read_limited(file, settings.max_upload_mb * 1024 * 1024)
    _check_content(filename, data)

    upload_dir = Path(settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename
    staged = upload_dir / f".{filename}.uploading"
    backup = upload_dir / f".{filename}.previous"
    replaced = target.exists()

    staged.write_bytes(data)
    _check_extractable(staged, filename)
    if replaced:
        os.replace(target, backup)
    os.replace(staged, target)
    try:
        summary = reindex(target, settings)
    except Exception:
        # Put the previous version back so the file on disk matches the index.
        if replaced:
            os.replace(backup, target)
        else:
            target.unlink(missing_ok=True)
        raise
    finally:
        staged.unlink(missing_ok=True)
    backup.unlink(missing_ok=True)

    request.app.state.corpus_version = summary.corpus_version
    request.state.log_fields.update(
        {"upload_bytes": len(data), "document_id": summary.document_id, "chunks": summary.chunks}
    )
    return UploadResult(
        document_id=summary.document_id,
        filename=filename,
        chunks=summary.chunks,
        pages=summary.pages,
        replaced_previous_version=replaced,
    )


@router.get("", response_model=list[DocumentInfo])
def list_documents(request: Request, key_id: str = Depends(enforce_rate_limit)) -> list[DocumentInfo]:
    settings = get_settings()
    chunks = get_indexed_chunks(get_qdrant(settings), settings)
    grouped: dict[str, dict] = defaultdict(lambda: {"chunks": 0, "pages": 0, "filename": ""})
    for chunk in chunks:
        meta = chunk.metadata
        document_id = str(meta.get("document_id", "unknown"))
        entry = grouped[document_id]
        entry["chunks"] += 1
        entry["filename"] = Path(str(meta.get("source", ""))).name
        entry["pages"] = max(entry["pages"], int(meta.get("page_end_number") or meta.get("page_number") or 1))
    request.state.log_fields["documents"] = len(grouped)
    return sorted(
        (DocumentInfo(document_id=doc_id, **entry) for doc_id, entry in grouped.items()),
        key=lambda doc: (doc.filename, doc.document_id),
    )


@router.delete("/{document_id}", response_model=DeleteResult)
def remove_document(request: Request, document_id: str, key_id: str = Depends(enforce_rate_limit)) -> DeleteResult:
    if not _DOCUMENT_ID.match(document_id):
        raise HTTPException(status_code=422, detail="Invalid document id.")
    settings = get_settings()
    client = get_qdrant(settings)
    existing = count_document_chunks(client, settings, document_id)
    if existing == 0:
        raise HTTPException(status_code=404, detail="No document with that id.")

    sources = {
        str(chunk.metadata.get("source"))
        for chunk in get_indexed_chunks(client, settings)
        if chunk.metadata.get("document_id") == document_id
    }
    delete_document(client, settings, document_id)

    # Remove the stored upload too, but never touch files outside UPLOAD_DIR.
    upload_dir = Path(settings.upload_dir).resolve()
    for source in sources:
        path = Path(source).resolve()
        if path.parent == upload_dir and path.is_file():
            path.unlink()

    _corpus_changed(request)
    request.state.log_fields.update({"document_id": document_id, "chunks": existing})
    return DeleteResult(document_id=document_id, deleted_chunks=existing)
