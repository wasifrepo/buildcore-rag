"""Compliance checklist chunker for BuildCore inspection and audit documents.

Two checklist types appear in the BuildCore corpus:

**SSIC-001** — Daily Site Safety Inspection Checklist
    Sections are identified by a letter (``SECTION A — SITE ACCESS AND TRAFFIC
    MANAGEMENT``).  Item codes follow the pattern ``A01``, ``A02`` (one letter
    + two digits).  The document closes with an ``OVERALL INSPECTION RESULT``
    summary section and a ``CORRECTIVE ACTION LOG (complete for each FAIL)``
    log section.

**SC-PMCL-001** — Subcontractor Pre-Mobilisation Compliance Checklist
    Sections are identified by a number (``SECTION 1 — LEGAL AND INSURANCE
    COMPLIANCE``).  Item codes follow the pattern ``1.01``, ``1.02`` (one or
    more digits + dot + two digits).  The document closes with an
    ``OVERALL RESULT`` summary section; no corrective action log is present.

Some sections in both checklists have no item-code rows — for example,
``SECTION 3 — WORKER CREDENTIALS`` in SC-PMCL-001 (worker name rows rather
than coded items) and the summary / log sections.  ``item_count`` is ``0``
for these sections.

Chunking strategy
-----------------
1. The document header block (before the first ━ line) becomes a single chunk
   carrying the parsed key-value fields (Form No., Project, Site Address,
   etc.) as metadata.
2. Each ━-delimited section becomes exactly one chunk — sections are never
   split.  Tabular rows bounded by ─ separator lines must stay together within
   their section, and the section boundary guarantee makes this automatic.
3. Special trailing sections (``OVERALL INSPECTION RESULT``, ``OVERALL
   RESULT``, ``CORRECTIVE ACTION LOG``) that have no ``SECTION X —`` header
   are included as single chunks with an empty ``section_letter``.

Metadata per chunk
------------------
``section_letter``           — section identifier string: a single letter
                               (``"A"``) for letter-based checklists, a
                               number string (``"1"``) for number-based
                               checklists, or ``""`` for the header block
                               and special trailing sections
``section_title``            — section title string
                               (``"SITE ACCESS AND TRAFFIC MANAGEMENT"``)
                               or ``"Document Header"`` for the header chunk
``item_count``               — number of checklist item rows detected in the
                               section body by the item code pattern
``has_corrective_action_log``— ``True`` when the chunk contains the
                               ``CORRECTIVE ACTION LOG`` section; only ever
                               ``True`` for one chunk per document
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Module-level compiled patterns
# ---------------------------------------------------------------------------

# Section delimiter: a line consisting entirely of ━ (U+2501 heavy horizontal)
_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^━+$")

# Section header — handles both corpus formats:
#   Letter-based: "SECTION A — SITE ACCESS AND TRAFFIC MANAGEMENT"
#   Number-based: "SECTION 1 — LEGAL AND INSURANCE COMPLIANCE"
# Captures the identifier (letter or digit) and the title.
_SECTION_HEADER: re.Pattern[str] = re.compile(
    r"^SECTION ([A-Z0-9]+)\s+—\s+(.+)$"
)

# Checklist item code — matches both corpus formats at the start of a line:
#   Letter format:  "A01   Site entry/exit gates secured …"
#   Numeric format: "1.01   Signed Subcontractor Services Agreement …"
# The \s after the code ensures a separator column follows (guards against
# false positives from header row labels or body text).
_ITEM_CODE: re.Pattern[str] = re.compile(
    r"^([A-Z]\d{2}|\d+\.\d{2})\s", re.MULTILINE
)

# Substring that identifies the corrective action log section title.
_CORRECTIVE_ACTION_MARKER: str = "CORRECTIVE ACTION LOG"

# Key-value field in the document header block.
# Key characters include letters, spaces, slashes, and dots
# (e.g. "Form No.", "Site Address:", "Subcontractor Company:").
_HEADER_FIELD: re.Pattern[str] = re.compile(
    r"^([A-Za-z][A-Za-z. /]+):\s+(.+)$"
)


class ChecklistChunker(BaseChunker):
    """Chunker for BuildCore compliance checklist documents.

    Splits a checklist into one chunk per ━-delimited section, preserving
    tabular item rows and section result lines within their parent section
    so that the full structured content is always co-embedded.
    """

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split a compliance checklist into section-aware Chunk objects.

        Produces one chunk for the document header block
        (``section_letter=""``, ``section_title="Document Header"``),
        then one chunk per ━-delimited section.  Chunk content always
        begins with the section heading so each chunk is self-contained
        for embedding and citation display.

        Args:
            content: Full raw text of the checklist file as read from disk.
            metadata: Must contain ``document_id``, ``document_type``, and
                ``source_path``.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects
            reflecting the document's section structure, one per section
            plus a leading header chunk.
        """
        document_id: str = metadata["document_id"]
        document_type: str = metadata["document_type"]
        source_path: str = metadata["source_path"]

        text = self.clean_whitespace(content)
        segments = self._split_on_separators(text)

        # segments[0]           = document header block
        # segments[1], [3], ... = section title strings  (odd indices)
        # segments[2], [4], ... = section body strings   (even indices >= 2)
        header_block = segments[0]

        chunks: list[Chunk] = []
        chunk_index = 0

        # --- Chunk 0: document header ----------------------------------------
        if header_block.strip():
            header_fields = self._parse_header_fields(header_block)
            chunks.append(
                self.build_chunk(
                    content=header_block,
                    document_id=document_id,
                    document_type=document_type,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    extra_metadata={
                        "section_letter": "",
                        "section_title": "Document Header",
                        "item_count": 0,
                        "has_corrective_action_log": False,
                        **header_fields,
                    },
                )
            )
            chunk_index += 1

        # --- One chunk per section -------------------------------------------
        i = 1
        while i + 1 < len(segments):
            section_title_raw = segments[i].strip()
            section_body = segments[i + 1]
            i += 2

            if not section_body.strip():
                continue

            section_letter, section_title = self._parse_section_header(
                section_title_raw
            )

            item_count = self._count_items(section_body)
            is_corrective_log = _CORRECTIVE_ACTION_MARKER in section_title_raw

            heading = (
                f"SECTION {section_letter} — {section_title}"
                if section_letter
                else section_title_raw
            )
            chunk_content = f"{heading}\n\n{section_body.strip()}"

            chunks.append(
                self.build_chunk(
                    content=chunk_content,
                    document_id=document_id,
                    document_type=document_type,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    extra_metadata={
                        "section_letter": section_letter,
                        "section_title": section_title,
                        "item_count": item_count,
                        "has_corrective_action_log": is_corrective_log,
                    },
                )
            )
            chunk_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_on_separators(text: str) -> list[str]:
        """Split the full document text on ━-only separator lines.

        Iterates line by line and begins a new segment each time a ━
        separator line is encountered.  The resulting list follows the
        same interleaving structure used by all other BuildCore chunkers:

        - ``segments[0]``  — document header block
        - ``segments[1]``, ``[3]``, ... — section title strings (odd indices)
        - ``segments[2]``, ``[4]``, ... — section body strings (even indices >= 2)

        The ─ table separator lines within section bodies are not affected
        by this split — only ━ lines trigger a new segment.

        Args:
            text: Whitespace-normalised full document text.

        Returns:
            Ordered list of text segments between ━ separator lines.
        """
        lines = text.split("\n")
        segments: list[str] = []
        current: list[str] = []

        for line in lines:
            if _SEPARATOR_LINE.match(line):
                segments.append("\n".join(current))
                current = []
            else:
                current.append(line)

        if current:
            segments.append("\n".join(current))

        return segments

    @staticmethod
    def _parse_section_header(raw: str) -> tuple[str, str]:
        """Parse a section title string into ``(section_letter, section_title)``.

        Handles both corpus section header formats:

        - ``"SECTION A — SITE ACCESS AND TRAFFIC MANAGEMENT"``
          → ``("A", "SITE ACCESS AND TRAFFIC MANAGEMENT")``
        - ``"SECTION 1 — LEGAL AND INSURANCE COMPLIANCE"``
          → ``("1", "LEGAL AND INSURANCE COMPLIANCE")``
        - ``"OVERALL INSPECTION RESULT"`` (no SECTION prefix)
          → ``("", "OVERALL INSPECTION RESULT")``
        - ``"CORRECTIVE ACTION LOG (complete for each FAIL)"``
          → ``("", "CORRECTIVE ACTION LOG (complete for each FAIL)")``

        Args:
            raw: Stripped section title string extracted from between ━ lines.

        Returns:
            Two-tuple of ``(section_letter, section_title)``.
            ``section_letter`` is the section identifier (single letter or
            number string) or an empty string for unnumbered/special sections.
        """
        m = _SECTION_HEADER.match(raw)
        if m:
            return m.group(1), m.group(2).strip()
        return "", raw

    @staticmethod
    def _count_items(body: str) -> int:
        """Count the number of checklist item rows in a section body.

        Scans for lines that begin with a recognised item code followed by
        whitespace.  Handles both corpus item code formats:

        - Letter format:  ``A01``, ``B03``, ``G08`` (one letter + two digits)
        - Numeric format: ``1.01``, ``2.06``, ``5.04`` (digits + dot + two digits)

        Lines that form table header rows, result summary lines, or free-text
        body paragraphs do not match either pattern and are excluded.

        Args:
            body: Raw section body text (content between the ━ separator pair,
                not including the section title line).

        Returns:
            Integer count of matched item-code lines.  Returns ``0`` for
            sections that contain no coded checklist items (e.g. worker
            credential rows, overall result tables, corrective action logs).
        """
        return len(_ITEM_CODE.findall(body))

    @staticmethod
    def _parse_header_fields(header: str) -> dict[str, str]:
        """Parse key: value metadata fields from the document header block.

        Extracts structured fields such as Form No., Project, Site Address,
        Subcontractor Company, and Mobilisation Date.  Keys are normalised
        by lowercasing, removing dots, and replacing spaces and slashes with
        underscores (e.g. ``"Form No."`` → ``"form_no"``,
        ``"Site Address"`` → ``"site_address"``).

        Only lines matching the ``KEY:  value`` pattern are extracted;
        free-text instruction lines and blank-field rows
        (``"Inspector Name:   ________________"``) that contain only
        underscores as values are included as-is.

        Args:
            header: Raw document header block text (the first segment before
                the first ━ separator line).

        Returns:
            Dict of normalised field names to their string values.  Empty dict
            if no key-value fields are found.
        """
        fields: dict[str, str] = {}
        for line in header.split("\n"):
            m = _HEADER_FIELD.match(line.strip())
            if m:
                key = (
                    m.group(1)
                    .strip()
                    .lower()
                    .replace(".", "")
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                fields[key] = m.group(2).strip()
        return fields
