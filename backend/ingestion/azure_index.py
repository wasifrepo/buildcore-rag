"""Azure AI Search index definition and child-chunk indexing (push model).

This is the Azure counterpart to the ChromaDB path in ``ingestion.pipeline``.
It creates the search index and uploads the *child* chunks produced by the
structure-aware chunkers plus ``ingestion.child_splitter``.

Why the push model, not index projections
-----------------------------------------
Azure AI Search can chunk documents for you via an indexer + skillset, with
"index projections" fanning parents out into child documents.  BuildCore does
**not** use that path, deliberately.  The built-in ``SplitSkill`` splits on
fixed character/page boundaries — precisely the naive chunking this project
exists to avoid.  Adopting it would discard the five document-type-aware
chunkers that are the heart of the ingestion design.

Instead we keep chunking in Python, embed the children ourselves, and *push*
finished documents into the index.  Azure AI Search is then a pure retrieval
engine (vector + BM25 + semantic ranking) rather than a chunking engine, and
the local and Azure backends index byte-identical child text — which is what
makes the evaluation harness a fair comparison between them.

Index shape
-----------
One search document per **child** chunk.  Each child carries its parent's ID
and full text, mirroring ``ingestion.child_splitter``'s ChromaDB metadata, so
``retrieval.azure_ai_search_retriever`` can collapse children back to parents
exactly as ``retrieval._parenting`` does locally.

The ``parent_content`` field is **retrievable but not searchable**.  This is
load-bearing: making it searchable would let BM25 match against full parent
text and silently destroy the small-to-big property, since matching would no
longer happen exclusively against small, focused children.

Heterogeneous structural metadata (``section_title``, ``clause_id``,
``sender``, ``step_number`` …) differs per document type, and Azure AI Search
requires a fixed schema.  Rather than declare every field across five document
types, the remaining metadata is stored as a single JSON string in
``metadata_json`` and rehydrated at retrieval time.

Configuration (environment variables)
-------------------------------------
* ``AZURE_SEARCH_ENDPOINT``           — ``https://<name>.search.windows.net``.
* ``AZURE_SEARCH_INDEX``              — index name (default ``"buildcore"``).
* ``AZURE_SEARCH_API_KEY``            — admin key; omit to use Managed Identity.
* ``AZURE_SEARCH_SEMANTIC_CONFIG``    — semantic configuration name to create.
* ``AZURE_SEARCH_VECTOR_FIELD``       — child embedding field name.
* ``AZURE_SEARCH_PARENT_ID_FIELD``    — parent key field name.
* ``AZURE_SEARCH_VECTOR_DIMENSIONS``  — embedding width (default ``1536``).
"""

import json
import logging
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    IncompleteReadError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.llm_client import embed_texts, get_embedding_model
from generation.schemas import Chunk
from ingestion.child_splitter import build_child_chunks
from ingestion.pipeline import ingest_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
_DEFAULT_DATA_DIR: Path = (
    Path(os.environ["DATA_DIR"])
    if "DATA_DIR" in os.environ
    else _PROJECT_ROOT / "data" / "raw"
)

_DEFAULT_INDEX_NAME: str = "buildcore"
#: text-embedding-3-small produces 1536-dimensional vectors.
_DEFAULT_VECTOR_DIMENSIONS: int = 1536

#: Chunks per OpenAI embeddings call.  Matches the ChromaDB path's batch size.
_EMBED_BATCH_SIZE: int = 100
#: Documents per Azure AI Search upload call.  The service caps a single
#: indexing batch at 1,000 documents (and 16 MB); 500 keeps payloads
#: comfortably clear of the size ceiling given each child carries its parent's
#: full text.
_UPLOAD_BATCH_SIZE: int = 500

_HNSW_CONFIG_NAME: str = "hnsw-cosine"
_VECTOR_PROFILE_NAME: str = "buildcore-vector-profile"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _require_endpoint() -> str:
    """Return the configured Azure AI Search endpoint.

    Returns:
        The service endpoint URL.

    Raises:
        ValueError: If ``AZURE_SEARCH_ENDPOINT`` is not set.
    """
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT is not set (e.g. "
            "https://<name>.search.windows.net)."
        )
    return endpoint


def get_index_name() -> str:
    """Return the target index name.

    Returns:
        ``AZURE_SEARCH_INDEX`` if set, else ``"buildcore"``.
    """
    return os.environ.get("AZURE_SEARCH_INDEX", _DEFAULT_INDEX_NAME)


def get_vector_field() -> str:
    """Return the child embedding field name.

    Returns:
        ``AZURE_SEARCH_VECTOR_FIELD`` if set, else ``"contentVector"``.
    """
    return os.environ.get("AZURE_SEARCH_VECTOR_FIELD", "contentVector")


def get_parent_id_field() -> str:
    """Return the parent key field name.

    Returns:
        ``AZURE_SEARCH_PARENT_ID_FIELD`` if set, else ``"parent_id"``.
    """
    return os.environ.get("AZURE_SEARCH_PARENT_ID_FIELD", "parent_id")


def get_vector_dimensions() -> int:
    """Return the embedding width the index expects.

    Returns:
        ``AZURE_SEARCH_VECTOR_DIMENSIONS`` if set, else ``1536``.
    """
    return int(
        os.environ.get("AZURE_SEARCH_VECTOR_DIMENSIONS", _DEFAULT_VECTOR_DIMENSIONS)
    )


def build_credential() -> AzureKeyCredential | object:
    """Return the credential used for Azure AI Search calls.

    Uses ``AZURE_SEARCH_API_KEY`` when set, otherwise ``DefaultAzureCredential``
    (Managed Identity in Azure, developer credentials locally).  The key path
    exists for laptop-to-cloud work; the identity path is the production
    configuration and keeps long-lived secrets out of the deployment.

    Returns:
        An :class:`~azure.core.credentials.AzureKeyCredential` or a
        ``DefaultAzureCredential`` token credential.

    Raises:
        ImportError: If Managed Identity is selected but ``azure-identity`` is
            not installed.
    """
    api_key = os.environ.get("AZURE_SEARCH_API_KEY")
    if api_key:
        return AzureKeyCredential(api_key)

    try:
        from azure.identity import DefaultAzureCredential  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "azure-identity is required when AZURE_SEARCH_API_KEY is unset. "
            "Install it, or set the key to use key-based auth."
        ) from exc
    return DefaultAzureCredential()


def get_search_client() -> SearchClient:
    """Build a query/upload client bound to the configured index.

    Returns:
        A configured :class:`~azure.search.documents.SearchClient`.
    """
    return SearchClient(
        endpoint=_require_endpoint(),
        index_name=get_index_name(),
        credential=build_credential(),
    )


#: Transient Azure AI Search failures worth retrying.
#:
#: IncompleteReadError is listed explicitly rather than catching its parent.
#: Its MRO is DecodeError -> HttpResponseError -> AzureError, so retrying
#: HttpResponseError would also retry a 400 Bad Request or a 404 — permanent
#: errors that fail identically on every attempt and should surface at once.
_SEARCH_RETRYABLE = (
    ServiceRequestError,
    ServiceResponseError,
    IncompleteReadError,
)


@retry(
    retry=retry_if_exception_type(_SEARCH_RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    reraise=True,
)
def search_with_retry(**kwargs) -> list[dict]:
    """Run a search query and materialise the results, retrying transient faults.

    ``SearchClient.search`` returns a lazy paged iterator: the HTTP response
    body is streamed as the caller iterates, so a dropped connection surfaces
    during *iteration*, not at call time.  Retrying only the ``search`` call
    would therefore not protect against the common failure —
    ``IncompleteReadError`` mid-stream.  This helper materialises the results
    inside the retried scope so a broken read is retried as one unit.

    Args:
        **kwargs: Passed through unchanged to
            :meth:`azure.search.documents.SearchClient.search`.

    Returns:
        The result rows as plain dicts, fully read from the wire.

    Raises:
        azure.core.exceptions.AzureError: If every attempt fails, the last
            error is re-raised.
    """
    results = get_search_client().search(**kwargs)
    return [dict(row) for row in results]


def get_index_client() -> SearchIndexClient:
    """Build a client for index management (create/update/delete).

    Returns:
        A configured :class:`~azure.search.documents.indexes.SearchIndexClient`.
    """
    return SearchIndexClient(
        endpoint=_require_endpoint(),
        credential=build_credential(),
    )


# ---------------------------------------------------------------------------
# Index definition
# ---------------------------------------------------------------------------


def build_index_definition() -> SearchIndex:
    """Construct the BuildCore child-chunk index definition.

    Declares the field schema, an HNSW vector profile using cosine distance
    (matching the local ChromaDB collection's ``hnsw:space`` setting and the
    unit-normalised OpenAI embeddings), and a semantic configuration that ranks
    on the child ``content`` field.

    Semantic ranking targets children rather than parents on purpose: the
    semantic ranker truncates long inputs, and children are already small and
    topically focused, which is exactly the input it performs best on.

    Returns:
        A :class:`~azure.search.documents.indexes.models.SearchIndex` ready to
        pass to :meth:`SearchIndexClient.create_or_update_index`.
    """
    vector_field = get_vector_field()
    parent_id_field = get_parent_id_field()

    fields = [
        # Child chunk ID. Azure restricts key characters to letters, digits,
        # dashes, underscores and equals signs; the chunkers' SHA-256-derived
        # hex IDs satisfy this.
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
        ),
        # The child text: the only field BM25 searches.
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name=vector_field,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=get_vector_dimensions(),
            vector_search_profile_name=_VECTOR_PROFILE_NAME,
        ),
        SimpleField(
            name=parent_id_field,
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Retrievable but NOT searchable — see module docstring. Making this
        # searchable would reintroduce parent-level matching and defeat
        # small-to-big retrieval.
        SimpleField(
            name="parent_content",
            type=SearchFieldDataType.String,
        ),
        SimpleField(name="parent_index", type=SearchFieldDataType.Int32),
        SimpleField(name="child_index", type=SearchFieldDataType.Int32),
        SimpleField(
            name="document_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="document_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # Per-document-type structural metadata, JSON-encoded.
        SimpleField(name="metadata_json", type=SearchFieldDataType.String),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=_HNSW_CONFIG_NAME,
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=_VECTOR_PROFILE_NAME,
                algorithm_configuration_name=_HNSW_CONFIG_NAME,
            )
        ],
    )

    semantic_config_name = os.environ.get(
        "AZURE_SEARCH_SEMANTIC_CONFIG", "buildcore-semantic"
    )
    semantic_search = SemanticSearch(
        default_configuration_name=semantic_config_name,
        configurations=[
            SemanticConfiguration(
                name=semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="document_type")],
                ),
            )
        ],
    )

    return SearchIndex(
        name=get_index_name(),
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_or_update_index() -> str:
    """Create the BuildCore index, or update it in place if it already exists.

    Returns:
        The name of the created/updated index.
    """
    index_client = get_index_client()
    index = build_index_definition()
    index_client.create_or_update_index(index)
    logger.info("Index '%s' created/updated.", index.name)
    return index.name


def delete_index() -> None:
    """Delete the BuildCore index if it exists.

    Field schema changes (such as altering the vector width) cannot always be
    applied in place; dropping and recreating is the reliable path for a corpus
    this small, where re-indexing costs cents and seconds.
    """
    index_client = get_index_client()
    index_client.delete_index(get_index_name())
    logger.info("Index '%s' deleted.", get_index_name())


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_corpus(
    data_dir: str | Path | None = None,
    recreate: bool = False,
) -> dict[str, int]:
    """Chunk, embed, and push the whole corpus into Azure AI Search.

    Mirrors :func:`ingestion.pipeline.run_ingestion`, but targets Azure AI
    Search instead of ChromaDB.  Both paths share the same chunkers and the
    same child splitter, so the two backends index identical text.

    Uploads use ``upload_documents``, which is an upsert keyed on the child's
    deterministic ID.  Re-running against an unchanged corpus therefore
    replaces each document with an identical copy rather than duplicating it,
    preserving the idempotence of the ChromaDB path.

    Files that fail to chunk, embed, or upload are logged and skipped so one
    bad document cannot abort the run.

    Args:
        data_dir: Root directory to walk.  Defaults to ``DATA_DIR`` or
            ``<project_root>/data/raw``.
        recreate: When ``True``, delete the index before recreating it.  Use
            after changing the field schema or the embedding model.

    Returns:
        Dict with ``files_processed``, ``files_failed``, ``parents``, and
        ``children_indexed`` counts.
    """
    resolved_data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    if recreate:
        try:
            delete_index()
        except Exception:  # noqa: BLE001 - absent index is not an error here
            logger.info("No existing index to delete; creating fresh.")
    create_or_update_index()

    search_client = get_search_client()
    embedding_model = get_embedding_model()

    corpus_files = sorted(
        [*resolved_data_dir.rglob("*.txt"), *resolved_data_dir.rglob("*.pdf")]
    )
    logger.info(
        "Indexing %d file(s) from '%s' into '%s'.",
        len(corpus_files),
        resolved_data_dir,
        get_index_name(),
    )

    files_processed = 0
    files_failed = 0
    total_parents = 0
    total_children = 0

    for file_path in corpus_files:
        logger.info("Processing '%s'", file_path.name)
        try:
            parents = ingest_file(file_path)
            children = [
                child for parent in parents for child in build_child_chunks(parent)
            ]
            _embed_and_upload(children, search_client, embedding_model)
            files_processed += 1
            total_parents += len(parents)
            total_children += len(children)
            logger.info(
                "  ✓ %d parent chunks → %d child chunks indexed from '%s'",
                len(parents),
                len(children),
                file_path.name,
            )
        except Exception:
            files_failed += 1
            logger.exception("  ✗ Failed to index '%s'", file_path.name)

    logger.info(
        "Indexing complete: %d file(s) processed, %d failed, %d children indexed.",
        files_processed,
        files_failed,
        total_children,
    )
    return {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "parents": total_parents,
        "children_indexed": total_children,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _embed_and_upload(
    children: list[Chunk],
    search_client: SearchClient,
    embedding_model: str,
) -> None:
    """Embed child chunks and upload them to Azure AI Search in batches.

    Embedding goes through :func:`common.llm_client.embed_texts`, which retries
    transient failures — without it a single dropped connection loses the whole
    document from the index.

    Args:
        children: Child chunks from :func:`ingestion.child_splitter.build_child_chunks`.
        search_client: Client bound to the target index.
        embedding_model: Embedding model or Azure deployment name.

    Raises:
        RuntimeError: If Azure reports a failure for any uploaded document.
    """
    documents: list[dict] = []
    for batch_start in range(0, len(children), _EMBED_BATCH_SIZE):
        batch = children[batch_start : batch_start + _EMBED_BATCH_SIZE]
        vectors = embed_texts(
            [child.content for child in batch], model=embedding_model
        )
        for child, vector in zip(batch, vectors):
            documents.append(_to_search_document(child, vector))

    for batch_start in range(0, len(documents), _UPLOAD_BATCH_SIZE):
        batch = documents[batch_start : batch_start + _UPLOAD_BATCH_SIZE]
        results = search_client.upload_documents(documents=batch)
        failures = [r for r in results if not r.succeeded]
        if failures:
            raise RuntimeError(
                f"{len(failures)} document(s) failed to index; first error: "
                f"key={failures[0].key} status={failures[0].status_code} "
                f"message={failures[0].error_message}"
            )


def _to_search_document(child: Chunk, embedding: list[float]) -> dict:
    """Convert a child chunk plus its embedding into an Azure search document.

    Pulls the parent linkage fields written by
    :func:`ingestion.child_splitter.build_child_chunks` out of the child's
    metadata into first-class index fields, and JSON-encodes whatever
    structural metadata remains.

    Args:
        child: The child chunk to convert.
        embedding: The child content's embedding vector.

    Returns:
        A dict whose keys match the index field schema.
    """
    metadata = dict(child.metadata)
    parent_id = metadata.pop("parent_id", child.chunk_id)
    parent_content = metadata.pop("parent_content", child.content)
    parent_index = metadata.pop("parent_index", 0)
    child_index = metadata.pop("child_index", 0)

    return {
        "id": child.chunk_id,
        "content": child.content,
        get_vector_field(): embedding,
        get_parent_id_field(): parent_id,
        "parent_content": parent_content,
        "parent_index": int(parent_index),
        "child_index": int(child_index),
        "document_id": child.document_id,
        "document_type": str(child.document_type),
        "metadata_json": json.dumps(metadata, default=str),
    }
