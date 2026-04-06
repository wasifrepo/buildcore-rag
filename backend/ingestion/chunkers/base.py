"""Abstract base class for all BuildCore document chunkers.

Each of the five document types in the corpus (safety SOPs, contracts,
incident emails, maintenance manuals, compliance checklists) has a
dedicated chunker subclass that implements the ``chunk`` abstract method.
All subclasses inherit the shared utility methods defined here so that
chunk ID generation, whitespace normalisation, and Chunk assembly are
consistent across the entire ingestion pipeline.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path

from generation.schemas import Chunk


class BaseChunker(ABC):
    """Abstract base class that all type-specific chunkers must extend.

    Subclasses must implement :meth:`chunk`. Everything else is shared
    infrastructure: whitespace normalisation, deterministic ID generation,
    and the single ``build_chunk`` factory that assembles a complete
    :class:`~generation.schemas.Chunk` from its constituent parts.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split raw document text into an ordered list of Chunk objects.

        Args:
            content: Full raw text of the document as read from disk.
            metadata: Dict that must contain at minimum:

                - ``document_id`` (str): Unique identifier for the source
                  document, typically the filename stem.
                - ``document_type`` (str): The ``DocumentType`` enum value
                  (as its string representation) for this document.
                - ``source_path`` (str): Absolute or relative path to the
                  source file, stored for provenance.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects.
            The list must not be empty for a non-empty document.
        """

    # ------------------------------------------------------------------
    # Shared utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalise whitespace in a text string without altering structure.

        Performs the following transformations in order:

        1. Normalises line endings to ``\\n`` (handles Windows ``\\r\\n``
           and legacy Mac ``\\r``).
        2. Strips trailing whitespace from every line.
        3. Collapses runs of three or more consecutive blank lines down to
           exactly two blank lines, preserving paragraph breaks.
        4. Strips leading and trailing whitespace from the entire string.

        Structural separator characters used as section boundaries in the
        BuildCore corpus (``━``, ``─``, ``═``) are deliberately preserved
        so that type-specific chunkers can detect and split on them.

        Args:
            text: Raw text to normalise.

        Returns:
            Normalised text string.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def generate_chunk_id(document_id: str, content: str) -> str:
        """Generate a deterministic, collision-resistant chunk ID.

        Hashes the concatenation of ``document_id`` and ``content`` with
        SHA-256 and returns the first 16 hexadecimal characters. Properties:

        - **Deterministic**: the same document and content always produce
          the same ID, so re-ingesting an unchanged document is idempotent.
        - **Document-scoped**: identical text appearing in two different
          source documents produces different IDs because the document ID
          is included in the hash input.
        - **Compact**: 16 hex chars (64-bit prefix) is sufficient to avoid
          collisions within a corpus of this size.

        Args:
            document_id: Unique identifier for the source document.
            content: Chunk text content, ideally after whitespace cleaning.

        Returns:
            A 16-character lowercase hexadecimal string.
        """
        hash_input = f"{document_id}::{content}"
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def derive_document_id(source_path: str | Path) -> str:
        """Derive a stable document ID from a source file path.

        Returns the filename stem (filename without extension), which in
        the BuildCore corpus is already a unique, human-readable reference
        number (e.g. ``SOP-001-fall-protection``, ``INC-2024-002-laceration-lti``).
        Using the stem rather than a hash keeps document IDs readable in
        traces, citations, and the evaluation harness.

        Args:
            source_path: Path to the source document file.

        Returns:
            The filename stem, preserving its original casing.
        """
        return Path(source_path).stem

    def build_chunk(
        self,
        content: str,
        document_id: str,
        document_type: str,
        source_path: str,
        chunk_index: int,
        extra_metadata: dict | None = None,
    ) -> Chunk:
        """Assemble a fully populated Chunk from its constituent parts.

        This is the primary factory method that subclasses call for every
        chunk they produce. It handles whitespace cleaning, ID generation,
        and metadata assembly so that subclasses only need to focus on
        *where* to split the document.

        Standard metadata fields written to every chunk's ``metadata`` dict:

        - ``source_path`` (str): Path to the originating file.
        - ``chunk_index`` (int): Zero-based position of this chunk within
          the document, reflecting reading order.

        Type-specific fields (e.g. ``section_title``, ``sender``,
        ``clause_id``, ``step_number``) are supplied by the subclass via
        ``extra_metadata`` and are merged in alongside the standard fields.

        Args:
            content: Raw chunk text. Whitespace is cleaned before storage
                and before the chunk ID is computed.
            document_id: Unique identifier for the source document.
            document_type: ``DocumentType`` string value for this document.
            source_path: Path to the source file, stored for provenance.
            chunk_index: Zero-based position of this chunk in the document.
            extra_metadata: Optional dict of type-specific fields to merge
                into the chunk's ``metadata``. Keys in this dict will
                overwrite standard fields if there is a name collision, so
                subclasses should avoid using ``source_path`` or
                ``chunk_index`` as keys.

        Returns:
            A :class:`~generation.schemas.Chunk` instance with all fields
            populated, ready for embedding and storage in ChromaDB.
        """
        cleaned = self.clean_whitespace(content)
        chunk_id = self.generate_chunk_id(document_id, cleaned)
        metadata: dict = {
            "source_path": str(source_path),
            "chunk_index": chunk_index,
            **(extra_metadata or {}),
        }
        return Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_type=document_type,
            content=cleaned,
            metadata=metadata,
        )
