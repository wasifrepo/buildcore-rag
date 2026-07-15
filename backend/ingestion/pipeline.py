"""Ingestion pipeline for BuildCore corpus documents.

Provides two public entry points:

``ingest_file``
    Reads a single document file, classifies its type, chunks it with the
    appropriate type-specific chunker, and returns the ordered list of
    :class:`~generation.schemas.Chunk` objects.  No I/O to external services
    is performed — suitable for testing chunkers in isolation.

    For ``.pdf`` files, text is extracted via ``pypdf.PdfReader`` before the
    chunker is called.  For ``.txt`` files the file is read directly as UTF-8.

``run_ingestion``
    Walks the full ``data/raw/`` directory tree, processes every ``.txt`` and
    ``.pdf`` file via ``ingest_file``, splits each structure-aware *parent*
    chunk into 2-3 sentence *child* chunks (see ``ingestion.child_splitter``),
    embeds every child with OpenAI ``text-embedding-3-small``, and upserts the
    children into a ChromaDB persistent collection.  Idempotent: re-running
    against an unchanged corpus produces the same chunk IDs and ChromaDB
    silently skips unchanged entries.

Small-to-big indexing
---------------------
Children are the units that get embedded and BM25-indexed, so matching happens
against small, focused text.  Each child carries its parent's ID and full text
in metadata, and the retrieval layer resolves a child hit back to its parent
(see ``retrieval/_parenting.py``) before reranking and generation — precise
matching, full-context answers.  Because children are character-capped below
the embedding token limit, ingestion never truncates: no source text is dropped
on the way into the store.

ChromaDB metadata constraint
-----------------------------
ChromaDB only accepts scalar metadata values (``str``, ``int``, ``float``,
``bool``).  Chunk metadata fields that are lists (``recipients`` in email
chunks, ``subsections`` in SOP chunks) are JSON-serialised to strings before
upsert and must be deserialised by the retrieval layer if needed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pypdf
from common.llm_client import embed_texts, get_embedding_model
from generation.schemas import Chunk, DocumentType
from ingestion.classifier import classify_document
from ingestion.child_splitter import build_child_chunks
from ingestion.chunkers.base import BaseChunker
from ingestion.chunkers.checklist_chunker import ChecklistChunker
from ingestion.chunkers.contract_chunker import ContractChunker
from ingestion.chunkers.email_chunker import EmailChunker
from ingestion.chunkers.manual_chunker import ManualChunker
from ingestion.chunkers.regulatory_chunker import RegulatoryChunker
from ingestion.chunkers.sop_chunker import SOPChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Resolved project root: backend/ingestion/pipeline.py → three levels up
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# Default corpus directory; overridden by the data_dir argument to run_ingestion.
# DATA_DIR env var takes precedence so the Docker container can point to /app/data/raw
# without relying on the project-root path resolution above.
_DEFAULT_DATA_DIR: Path = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else _PROJECT_ROOT / "data" / "raw"

# Number of chunks sent in a single OpenAI embeddings API call.
# Kept well below the 2 048-input hard limit to bound memory and latency.
_EMBED_BATCH_SIZE: int = 100

# Singleton chunker instances — one per document type.
# Instantiated once at module load to avoid repeated object creation during
# bulk ingestion; all chunkers are stateless so sharing is safe.
_CHUNKER_MAP: dict[DocumentType, BaseChunker] = {
    DocumentType.SAFETY_SOP: SOPChunker(),
    DocumentType.CONTRACT: ContractChunker(),
    DocumentType.INCIDENT_EMAIL: EmailChunker(),
    DocumentType.MAINTENANCE_MANUAL: ManualChunker(),
    DocumentType.COMPLIANCE_CHECKLIST: ChecklistChunker(),
    DocumentType.REGULATORY_DOC: RegulatoryChunker(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_file(file_path: str | Path) -> list[Chunk]:
    """Read, classify, and chunk a single document file.

    Reads the file content as UTF-8 text, determines the document type via
    :func:`~ingestion.classifier.classify_document`, selects the matching
    chunker from :data:`_CHUNKER_MAP`, and returns the ordered list of
    :class:`~generation.schemas.Chunk` objects produced by that chunker.

    No embedding or database I/O is performed.  This function is the
    correct entry point for unit-testing individual chunkers against real
    corpus files.

    Args:
        file_path: Absolute or relative path to the ``.txt`` or ``.pdf``
            document file.

    Returns:
        Ordered list of :class:`~generation.schemas.Chunk` objects, one or
        more per document depending on the chunker's splitting strategy.

    Raises:
        ValueError: If the document type cannot be determined from the file
            path (propagated from :func:`~ingestion.classifier.classify_document`).
        FileNotFoundError: If the file does not exist at ``file_path``.
        UnicodeDecodeError: If a .txt file cannot be decoded as UTF-8.
    """
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        content = _extract_pdf_text(path)
    else:
        content = path.read_text(encoding="utf-8")

    doc_type: DocumentType = classify_document(path)
    chunker: BaseChunker = _CHUNKER_MAP[doc_type]

    document_id = BaseChunker.derive_document_id(path)
    metadata = {
        "document_id": document_id,
        "document_type": doc_type.value,
        "source_path": str(path),
    }

    return chunker.chunk(content, metadata)


def run_ingestion(
    data_dir: str | Path | None = None,
    chroma_persist_dir: str | None = None,
    chroma_collection_name: str | None = None,
) -> dict[str, int]:
    """Walk the corpus directory, embed every chunk, and upsert into ChromaDB.

    Processes all ``.txt`` and ``.pdf`` files found anywhere under ``data_dir``.  For each
    file, chunks are produced by :func:`ingest_file`, embedded in batches via
    the OpenAI Embeddings API, and upserted into the specified ChromaDB
    collection.  ChromaDB's ``upsert`` semantics make this operation
    idempotent: running against an unchanged corpus is safe and produces no
    duplicate entries.

    Files that raise exceptions during chunking or embedding are logged at
    ERROR level and skipped; processing continues with the remaining files.

    Args:
        data_dir: Root directory to walk for corpus files.  Defaults to the
            ``data/raw/`` directory relative to the project root.
        chroma_persist_dir: Filesystem path for ChromaDB's persisted storage.
            Defaults to the ``CHROMA_PERSIST_DIR`` environment variable, or
            ``<project_root>/data/chroma`` if the variable is not set.
        chroma_collection_name: Name of the ChromaDB collection to upsert
            into.  Defaults to the ``CHROMA_COLLECTION_NAME`` environment
            variable, or ``"buildcore"`` if the variable is not set.

    Returns:
        Summary dict with the following integer keys:

        - ``files_processed``: number of files successfully chunked and upserted
        - ``files_failed``: number of files that raised an exception
        - ``chunks_upserted``: total number of chunk embeddings written to ChromaDB
    """
    resolved_data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    resolved_persist_dir = chroma_persist_dir or os.environ.get(
        "CHROMA_PERSIST_DIR",
        str(_PROJECT_ROOT / "data" / "chroma"),
    )
    resolved_collection_name = chroma_collection_name or os.environ.get(
        "CHROMA_COLLECTION_NAME", "buildcore"
    )
    embedding_model = get_embedding_model()

    # Imported here rather than at module scope so that chunking (ingest_file)
    # and the Azure indexing path (ingestion.azure_index, which reuses it) do
    # not drag ChromaDB in. The Azure production image omits chromadb entirely.
    import chromadb  # noqa: PLC0415

    chroma_client = chromadb.PersistentClient(path=resolved_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=resolved_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    corpus_files = sorted(
        [*resolved_data_dir.rglob("*.txt"), *resolved_data_dir.rglob("*.pdf")]
    )
    logger.info(
        "Starting ingestion: %d files found under '%s'",
        len(corpus_files),
        resolved_data_dir,
    )

    files_processed = 0
    files_failed = 0
    chunks_upserted = 0

    for file_path in corpus_files:
        logger.info("Processing '%s'", file_path.name)
        try:
            parents = ingest_file(file_path)
            children = [
                child for parent in parents for child in build_child_chunks(parent)
            ]
            _upsert_chunks(children, collection, embedding_model)
            chunks_upserted += len(children)
            files_processed += 1
            logger.info(
                "  ✓ %d parent chunks → %d child chunks upserted from '%s'",
                len(parents),
                len(children),
                file_path.name,
            )
        except Exception:
            files_failed += 1
            logger.exception("  ✗ Failed to process '%s'", file_path.name)

    logger.info(
        "Ingestion complete — %d files processed, %d failed, %d chunks upserted",
        files_processed,
        files_failed,
        chunks_upserted,
    )
    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "chunks_upserted": chunks_upserted,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> str:
    """Extract and concatenate all page texts from a PDF file.

    Uses ``pypdf.PdfReader`` to iterate over every page and join the extracted
    text with newline separators.  Pages that yield no text (e.g. scanned
    image pages) contribute an empty string and do not interrupt the join.

    Args:
        path: Filesystem path to the ``.pdf`` file.

    Returns:
        Concatenated plain text of all pages, separated by ``\\n``.
    """
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _upsert_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    embedding_model: str,
) -> None:
    """Embed a list of child chunks in batches and upsert them into ChromaDB.

    Splits ``chunks`` into batches of :data:`_EMBED_BATCH_SIZE`, requests
    embeddings for each batch via :func:`common.llm_client.embed_texts` (which
    retries transient API failures), flattens each chunk's metadata dict to
    ChromaDB-compatible scalars, and calls ``collection.upsert`` for the batch.

    The full chunk content is both embedded and stored verbatim — nothing is
    truncated.  Child chunks are character-capped by
    ``ingestion.child_splitter`` well below the embedding model's token limit,
    so the embedding call never rejects an over-long input and no source text
    is ever dropped.

    Args:
        chunks: Ordered list of child :class:`~generation.schemas.Chunk`
            objects to embed and store.
        collection: The ChromaDB collection to upsert into.
        embedding_model: OpenAI embedding model identifier
            (e.g. ``"text-embedding-3-small"``).
    """
    for batch_start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [chunk.content for chunk in batch]
        ids = [chunk.chunk_id for chunk in batch]

        embeddings = embed_texts(texts, model=embedding_model)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[_flatten_metadata(chunk) for chunk in batch],
        )


def _flatten_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Produce a ChromaDB-compatible metadata dict from a Chunk.

    ChromaDB requires that all metadata values are scalars (``str``, ``int``,
    ``float``, or ``bool``).  This function copies the chunk's standard fields
    and its ``metadata`` dict, serialising any ``list`` values to JSON strings
    so that downstream retrieval code can deserialise them with
    ``json.loads`` when needed.

    The chunk's ``document_id`` and ``document_type`` are always included as
    top-level metadata keys so they are available as ChromaDB ``where``
    filter fields without needing to inspect the nested metadata dict.

    Args:
        chunk: The :class:`~generation.schemas.Chunk` whose metadata should
            be flattened.

    Returns:
        Dict whose values are all ChromaDB-scalar types.
    """
    flat: dict[str, str | int | float | bool] = {
        "document_id": chunk.document_id,
        "document_type": chunk.document_type,
    }
    for key, value in chunk.metadata.items():
        if isinstance(value, list):
            flat[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            # Fallback: coerce unexpected types to string to avoid ChromaDB errors
            flat[key] = str(value)
    return flat
