from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from ragforge.ingestion.cleaner import clean_documents
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_file(file_path: str | Path) -> Path:
    """
    Validate that a file exists, is a regular file, is non-empty,
    and has a supported extension.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"File is empty: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Normalize path identity so relative and absolute invocations of the same
    # file produce the same source metadata and deterministic document ID.
    return path.resolve()


# ---------------------------------------------------------------------------
# Document ID
# ---------------------------------------------------------------------------

def _default_document_id(source: str) -> str:
    """
    Generate a deterministic document ID from the document source.

    The same source always produces the same document ID.
    """

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def normalize_metadata(
    document: Document,
    *,
    source: str,
    document_id: str | None = None,
) -> Document:
    """
    Normalize metadata so downstream RAG components receive
    a consistent metadata structure.

    If document_id is not explicitly supplied, a deterministic
    ID is generated from the source.
    """

    metadata = dict(document.metadata)

    metadata["source"] = source

    # Use caller-provided ID when available.
    # Otherwise generate a deterministic ID from the source.
    metadata["document_id"] = (
        document_id
        if document_id
        else _default_document_id(source)
    )

    # Normalize page numbering.
    # PyPDFLoader generally gives a zero-based page number.
    if "page" in metadata:
        metadata["page"] = int(metadata["page"])
        metadata["page_number"] = metadata["page"] + 1

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# PDF Loader
# ---------------------------------------------------------------------------

def load_pdf(
    pdf_path: str | Path,
    *,
    document_id: str | None = None,
) -> list[Document]:
    """
    Load a PDF into one LangChain Document per page.
    """

    path = validate_file(pdf_path)

    logger.info(
        "Loading PDF: %s",
        path,
    )

    loader = PyPDFLoader(str(path))

    try:
        documents = loader.load()
    except Exception as exc:
        logger.exception(
            "Failed to load PDF: %s",
            path,
        )
        raise RuntimeError(
            f"Failed to load PDF: {path}"
        ) from exc

    source = str(path)

    documents = [
        normalize_metadata(
            document,
            source=source,
            document_id=document_id,
        )
        for document in documents
    ]

    logger.info(
        "Loaded %d pages from PDF: %s",
        len(documents),
        path,
    )

    return documents


# ---------------------------------------------------------------------------
# Text / Markdown Loader
# ---------------------------------------------------------------------------

def load_text(
    file_path: str | Path,
    *,
    document_id: str | None = None,
) -> list[Document]:
    """
    Load a plain-text or Markdown document.
    """

    path = validate_file(file_path)

    logger.info(
        "Loading text file: %s",
        path,
    )

    loader = TextLoader(
        str(path),
        encoding="utf-8",
    )

    try:
        documents = loader.load()
    except Exception as exc:
        logger.exception(
            "Failed to load text file: %s",
            path,
        )
        raise RuntimeError(
            f"Failed to load text file: {path}"
        ) from exc

    source = str(path)

    documents = [
        normalize_metadata(
            document,
            source=source,
            document_id=document_id,
        )
        for document in documents
    ]

    logger.info(
        "Loaded text file: %s",
        path,
    )

    return documents


# ---------------------------------------------------------------------------
# Directory Loader
# ---------------------------------------------------------------------------

def load_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
) -> list[Document]:
    """
    Load all supported documents from a directory.

    Supported:
        PDF
        TXT
        Markdown
    """

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise ValueError(
            f"Path is not a directory: {directory_path}"
        )

    logger.info(
        "Loading documents from directory: %s",
        directory_path,
    )

    documents: list[Document] = []

    # -----------------------------------------------------------------------
    # PDFs
    # -----------------------------------------------------------------------

    pdf_pattern = (
        "**/*.pdf"
        if recursive
        else "*.pdf"
    )

    pdf_loader = DirectoryLoader(
        str(directory_path),
        glob=pdf_pattern,
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    try:
        documents.extend(
            pdf_loader.load()
        )
    except Exception as exc:
        logger.exception(
            "Failed while loading PDFs from: %s",
            directory_path,
        )
        raise RuntimeError(
            f"Failed to load PDFs from {directory_path}"
        ) from exc

    # -----------------------------------------------------------------------
    # Text files
    # -----------------------------------------------------------------------

    text_pattern = (
        "**/*.txt"
        if recursive
        else "*.txt"
    )

    text_loader = DirectoryLoader(
        str(directory_path),
        glob=text_pattern,
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        },
        show_progress=True,
        use_multithreading=True,
    )

    try:
        documents.extend(
            text_loader.load()
        )
    except Exception as exc:
        logger.exception(
            "Failed while loading text files from: %s",
            directory_path,
        )
        raise RuntimeError(
            f"Failed to load text files from {directory_path}"
        ) from exc

    # -----------------------------------------------------------------------
    # Markdown
    # -----------------------------------------------------------------------

    markdown_pattern = (
        "**/*.md"
        if recursive
        else "*.md"
    )

    markdown_loader = DirectoryLoader(
        str(directory_path),
        glob=markdown_pattern,
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        },
        show_progress=True,
        use_multithreading=True,
    )

    try:
        documents.extend(
            markdown_loader.load()
        )
    except Exception as exc:
        logger.exception(
            "Failed while loading Markdown files from: %s",
            directory_path,
        )
        raise RuntimeError(
            f"Failed to load Markdown files from {directory_path}"
        ) from exc

    # -----------------------------------------------------------------------
    # Normalize metadata
    # -----------------------------------------------------------------------

    normalized_documents: list[Document] = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        normalized_documents.append(
            normalize_metadata(
                document,
                source=source,
            )
        )

    logger.info(
        "Loaded %d document objects from %s",
        len(normalized_documents),
        directory_path,
    )

    return normalized_documents


# ---------------------------------------------------------------------------
# Web Loader
# ---------------------------------------------------------------------------

def load_web(
    url: str,
) -> list[Document]:
    """
    Load a webpage.

    Web ingestion is kept separate from file ingestion because
    web documents require different validation and metadata handling.
    """

    if not url.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            f"Invalid URL: {url}"
        )

    logger.info(
        "Loading webpage: %s",
        url,
    )

    loader = WebBaseLoader(url)

    try:
        documents = loader.load()
    except Exception as exc:
        logger.exception(
            "Failed to load webpage: %s",
            url,
        )
        raise RuntimeError(
            f"Failed to load webpage: {url}"
        ) from exc

    documents = [
        normalize_metadata(
            document,
            source=url,
        )
        for document in documents
    ]

    logger.info(
        "Loaded %d document(s) from webpage: %s",
        len(documents),
        url,
    )

    return documents


# ---------------------------------------------------------------------------
# Generic File Loader
# ---------------------------------------------------------------------------

def load_file(
    file_path: str | Path,
    *,
    document_id: str | None = None,
) -> list[Document]:
    """
    Automatically select the appropriate loader
    based on file extension.
    """

    path = validate_file(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(
            path,
            document_id=document_id,
        )

    if extension in {".txt", ".md"}:
        return load_text(
            path,
            document_id=document_id,
        )

    raise ValueError(
        f"Unsupported file extension: {extension}"
    )


# ---------------------------------------------------------------------------
# Document Validation
# ---------------------------------------------------------------------------

def validate_documents(
    documents: Iterable[Document],
) -> list[Document]:
    """
    Remove empty documents and validate the LangChain
    Document structure.
    """

    valid_documents: list[Document] = []

    for index, document in enumerate(documents):

        if not isinstance(
            document,
            Document,
        ):
            logger.warning(
                "Skipping invalid document at index %d",
                index,
            )
            continue

        if not document.page_content.strip():
            logger.warning(
                "Skipping empty document at index %d",
                index,
            )
            continue

        valid_documents.append(document)

    logger.info(
        "Document validation: %d valid document(s)",
        len(valid_documents),
    )

    return valid_documents


# ---------------------------------------------------------------------------
# Public Ingestion Function
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str | Path,
    *,
    document_id: str | None = None,
) -> list[Document]:
    """
    Main entry point for file ingestion.

    Pipeline:

        file
          ↓
        validation
          ↓
        loader
          ↓
        metadata normalization
          ↓
        document validation
          ↓
        LangChain Documents
    """

    logger.info(
        "Starting ingestion: %s",
        file_path,
    )

    documents = load_file(
        file_path,
        document_id=document_id,
    )

    documents = validate_documents(
        documents
    )
    documents = clean_documents(documents)

    logger.info(
        "Finished ingestion: %s | %d document(s)",
        file_path,
        len(documents),
    )

    return documents


# ---------------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    documents = ingest_file(
        "data/test.pdf",
        document_id="test-document",
    )

    print(
        f"\nLoaded {len(documents)} document(s)\n"
    )

    for i, document in enumerate(
        documents,
        start=1,
    ):

        print("=" * 80)
        print(f"DOCUMENT {i}")
        print("=" * 80)

        print(
            "Content preview:\n"
            f"{document.page_content[:500]}\n"
        )

        print(
            "Metadata:\n"
            f"{document.metadata}\n"
        )