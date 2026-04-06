"""Ingestion pipeline for BuildCore corpus documents.

Provides two public entry points:

``ingest_file``
    Reads a single document file, classifies its type, chunks it with the
    appropriate type-specific chunker, and returns the ordered list of
    :class:`~generation.schemas.Chunk` objects.  No I/O to external services
    is performed — suitable for testing chunkers in isolation.

``run_ingestion``
    Walks the full ``data/raw/`` directory tree, processes every ``.txt``
    file via ``ingest_file``, embeds each chunk with OpenAI
    ``text-embedding-3-small``, and upserts the results into a ChromaDB
    persistent collection.  Idempotent: re-running against an unchanged
    corpus produces the same chunk IDs and ChromaDB silently skips
    unchanged entries.

ChromaDB metadata constraint
-----------------------------
ChromaDB only accepts scalar metadata values (``str``, ``int``, ``float``,
``bool``).  Chunk metadata fields that are lists (``recipients`` in email
chunks, ``subsections`` in SOP chunks) are JSON-serialised to strings before
upsert and must be deserialised by the retrieval layer if needed.
"""

import json
import logging
import os
from pathlib import Path

import chromadb
from openai import OpenAI

from generation.schemas import Chunk, DocumentType
from ingestion.classifier import classify_document
from ingestion.chunkers.base import BaseChunker
from ingestion.chunkers.checklist_chunker import ChecklistChunker
from ingestion.chunkers.contract_chunker import ContractChunker
from ingestion.chunkers.email_chunker import EmailChunker
from ingestion.chunkers.manual_chunker import ManualChunker
from ingestion.chunkers.sop_chunker import SOPChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Resolved project root: backend/ingestion/pipeline.py → three levels up
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# Default corpus directory; overridden by the data_dir argument to run_ingestion
_DEFAULT_DATA_DIR: Path = _PROJECT_ROOT / "data" / "raw"

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
        file_path: Absolute or relative path to the ``.txt`` document file.

    Returns:
        Ordered list of :class:`~generation.schemas.Chunk` objects, one or
        more per document depending on the chunker's splitting strategy.

    Raises:
        ValueError: If the document type cannot be determined from the file
            path (propagated from :func:`~ingestion.classifier.classify_document`).
        FileNotFoundError: If the file does not exist at ``file_path``.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    path = Path(file_path)
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

    Processes all ``.txt`` files found anywhere under ``data_dir``.  For each
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
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path=resolved_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=resolved_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    txt_files = sorted(resolved_data_dir.rglob("*.txt"))
    logger.info(
        "Starting ingestion: %d files found under '%s'",
        len(txt_files),
        resolved_data_dir,
    )

    files_processed = 0
    files_failed = 0
    chunks_upserted = 0

    for file_path in txt_files:
        logger.info("Processing '%s'", file_path.name)
        try:
            chunks = ingest_file(file_path)
            _upsert_chunks(chunks, collection, openai_client, embedding_model)
            chunks_upserted += len(chunks)
            files_processed += 1
            logger.info(
                "  ✓ %d chunks upserted from '%s'", len(chunks), file_path.name
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


def _upsert_chunks(
    chunks: list[Chunk],
    collection: chromadb.Collection,
    openai_client: OpenAI,
    embedding_model: str,
) -> None:
    """Embed a list of chunks in batches and upsert them into a ChromaDB collection.

    Splits ``chunks`` into batches of :data:`_EMBED_BATCH_SIZE`, requests
    embeddings for each batch from the OpenAI Embeddings API, flattens
    each chunk's metadata dict to ChromaDB-compatible scalars, and calls
    ``collection.upsert`` for the batch.

    Args:
        chunks: Ordered list of :class:`~generation.schemas.Chunk` objects
            to embed and store.
        collection: The ChromaDB collection to upsert into.
        openai_client: Authenticated :class:`openai.OpenAI` client instance.
        embedding_model: OpenAI embedding model identifier
            (e.g. ``"text-embedding-3-small"``).
    """
    for batch_start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
        texts = [chunk.content for chunk in batch]

        response = openai_client.embeddings.create(
            model=embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]

        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
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
