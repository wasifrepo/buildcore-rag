"""Maintenance manual chunker for BuildCore equipment service documents.

Maintenance manuals in the BuildCore corpus are structured with ━ separator
lines framing section headers (e.g. ``SECTION 1 — PRE-OPERATION INSPECTION``,
``SECTION 3 — OPERATING PROCEDURES``).  Some sections are unnumbered
(``IMPORTANT SAFETY NOTICE``, ``SAFETY WARNINGS``).

Within numbered sections, procedures are written as top-level ``STEP N`` lines
(two formats observed in the corpus):

- **Format A** — ``STEP N — Title`` on its own line, followed by indented
  sub-steps ``  N.M description``.
- **Format B** — ``STEP N: single-line instruction`` with no sub-steps.

A third pattern appears in the forklift SECTION 5 (Emergency Procedures):
sub-steps are written as ``  Step N:`` (indented, sentence-case) under
numbered sub-sections (``5.1``, ``5.2``, ``5.3``).  These are *not* top-level
STEPs; ``has_steps`` is ``False`` and ``step_count`` is ``0`` for that section.

Chunking strategy
-----------------
1. The document header block (before the first ━ line) becomes a single chunk
   carrying the parsed key-value fields (Equipment, Make/Model, Serial No.,
   etc.) as metadata.
2. Each ━-delimited section becomes exactly one chunk — sections are never
   split regardless of length, because numbered step sequences and their
   sub-steps must always stay together.
3. Unnumbered sections (``IMPORTANT SAFETY NOTICE``, ``SAFETY WARNINGS``) are
   included verbatim as single chunks with an empty ``section_number``.

Metadata per chunk
------------------
``section_number``  — top-level section number string (``"1"``, ``"2"``, …)
                      or ``""`` for unnumbered sections and the header block
``section_title``   — section title string (``"PRE-OPERATION INSPECTION"``, …)
``has_steps``       — ``True`` when the chunk body contains at least one
                      top-level ``STEP N`` line (col 0, upper-case)
``step_count``      — number of distinct top-level ``STEP N`` entries found
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Module-level compiled patterns
# ---------------------------------------------------------------------------

# Section delimiter: a line consisting entirely of ━ (U+2501 heavy horizontal)
_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^━+$")

# Numbered section header: "SECTION 1 — PRE-OPERATION INSPECTION"
# The — is U+2014 EM DASH, matching the corpus exactly.
_SECTION_NUMBERED: re.Pattern[str] = re.compile(r"^SECTION (\d+)\s+—\s+(.+)$")

# Top-level STEP line — matches both corpus formats at column 0 (no indent):
#   Format A: "STEP 1 — Approach and Inspect the Forklift"
#   Format B: "STEP 1: Verify that the generator is on a level surface."
# The \b word boundary prevents matching indented "  Step N:" sub-steps that
# appear inside the Emergency Procedures section of the forklift manual.
_TOP_LEVEL_STEP: re.Pattern[str] = re.compile(r"^STEP \d+\b", re.MULTILINE)

# Key-value field in the document header block.
# Key characters include letters, spaces, slashes, and dots
# (e.g. "Make/Model", "Serial No.", "Manual Ref:").
_HEADER_FIELD: re.Pattern[str] = re.compile(
    r"^([A-Za-z][A-Za-z. /]+):\s+(.+)$"
)


class ManualChunker(BaseChunker):
    """Chunker for BuildCore equipment maintenance manual documents.

    Splits a manual into one chunk per ━-delimited section, preserving the
    full section body (header line + all procedure steps and sub-steps) so
    that step sequences are never broken across chunk boundaries.
    """

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split a maintenance manual into section-aware Chunk objects.

        Produces one chunk for the document header block
        (``section_number=""``, ``section_title="Document Header"``),
        then one chunk per numbered or unnumbered section.  Chunk content
        always begins with the section heading so each chunk is
        self-contained for embedding and citation display.

        Args:
            content: Full raw text of the manual file as read from disk.
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
                        "section_number": "",
                        "section_title": "Document Header",
                        "has_steps": False,
                        "step_count": 0,
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

            section_number, section_title = self._parse_section_header(
                section_title_raw
            )

            has_steps, step_count = self._analyse_steps(section_body)

            heading = (
                f"SECTION {section_number} — {section_title}"
                if section_number
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
                        "section_number": section_number,
                        "section_title": section_title,
                        "has_steps": has_steps,
                        "step_count": step_count,
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
        separator line is encountered.  The resulting list has the same
        interleaving structure as the SOP and contract chunkers:

        - ``segments[0]``  — document header block
        - ``segments[1]``, ``[3]``, ... — section title strings (odd indices)
        - ``segments[2]``, ``[4]``, ... — section body strings (even indices >= 2)

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
        """Parse a section title string into ``(section_number, section_title)``.

        Handles both numbered sections
        (``"SECTION 3 — OPERATING PROCEDURES"`` → ``("3", "OPERATING PROCEDURES")``)
        and unnumbered sections
        (``"SAFETY WARNINGS"`` → ``("", "SAFETY WARNINGS")``).

        Args:
            raw: Stripped section title string extracted from between ━ lines.

        Returns:
            Two-tuple of ``(section_number, section_title)``.
            ``section_number`` is an empty string for unnumbered sections.
        """
        m = _SECTION_NUMBERED.match(raw)
        if m:
            return m.group(1), m.group(2).strip()
        return "", raw

    @staticmethod
    def _analyse_steps(body: str) -> tuple[bool, int]:
        """Count top-level ``STEP N`` entries in a section body.

        Scans the body for lines that start with ``STEP`` followed by a digit
        at column 0 (no leading whitespace).  This correctly excludes the
        indented ``  Step N:`` sub-steps that appear under the numbered
        sub-sections of the forklift Emergency Procedures section.

        Args:
            body: Raw section body text (content between the ━ separator pair,
                not including the section title line).

        Returns:
            Two-tuple of ``(has_steps, step_count)`` where ``has_steps`` is
            ``True`` when at least one top-level STEP line is found and
            ``step_count`` is the total number of such lines.
        """
        matches = _TOP_LEVEL_STEP.findall(body)
        step_count = len(matches)
        return step_count > 0, step_count

    @staticmethod
    def _parse_header_fields(header: str) -> dict[str, str]:
        """Parse key: value metadata fields from the document header block.

        Extracts structured fields such as Equipment, Make/Model, Serial No.,
        Unit ID, and Manual Ref.  Keys are normalised by lowercasing, removing
        dots, and replacing spaces and slashes with underscores so that
        downstream code can access them with consistent attribute-style names
        (e.g. ``"Make/Model"`` → ``"make_model"``,
        ``"Serial No."`` → ``"serial_no"``).

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
