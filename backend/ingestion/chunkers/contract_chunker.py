"""Contract document chunker for BuildCore subcontractor services agreements.

Contracts in the BuildCore corpus are structured with ━ separator lines
framing schedule headers (e.g. ``SCHEDULE A — SCOPE OF WORKS``) and, in
some contracts, a ``GENERAL CONDITIONS`` section containing numbered clauses.
Financial and insurance tables within schedules use ─ separator lines.

Chunking strategy
-----------------
1. The document header block (before the first ━ line) becomes a single chunk
   carrying the parsed agreement fields (Agreement No., parties, dates) as
   metadata.
2. Each ━-delimited schedule (``SCHEDULE A`` through ``SCHEDULE E``) becomes
   one chunk regardless of length.  Financial tables (rows bounded by ─ lines)
   must never be split — keeping the whole schedule together guarantees this.
3. ``GENERAL CONDITIONS``, when present as an explicit ━-delimited section, is
   subdivided into one chunk per ``Clause N — Title`` entry.  Any preamble
   before the first clause and the execution block that trails the last clause
   remain attached to their nearest clause chunk.  When no explicit GENERAL
   CONDITIONS section exists (e.g. Apex contract, where terms are referenced
   inline within Schedule D), no clause chunks are produced.

Every chunk carries ``schedule_name``, ``clause_id``, and ``has_table`` in
its metadata.  ``clause_id`` is a non-empty string only for chunks produced
from a GENERAL CONDITIONS section.  ``has_table`` is ``True`` when the chunk
body contains at least one ─-only separator line.
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Module-level compiled patterns
# ---------------------------------------------------------------------------

# Section delimiter: a line consisting entirely of ━ (U+2501 heavy horizontal)
_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^━+$")

# Table row separator: a line of ─ (U+2500 light horizontal), optional indent
_TABLE_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^\s*─+\s*$")

# GENERAL CONDITIONS clause header: "Clause 1 — Governing Law"
# The — is U+2014 EM DASH, matching the corpus exactly.
_CLAUSE_LINE: re.Pattern[str] = re.compile(r"^Clause (\d+)\s+—\s+(.+)$")

# Key-value field in the document header block.
# Key characters include letters, spaces, slashes, and dots (for "Agreement No.")
_HEADER_FIELD: re.Pattern[str] = re.compile(
    r"^([A-Za-z][A-Za-z. /]+):\s+(.+)$"
)

# The literal string that identifies the general conditions section title
_GENERAL_CONDITIONS_TITLE: str = "GENERAL CONDITIONS"


class ContractChunker(BaseChunker):
    """Chunker for BuildCore subcontractor services agreement documents.

    Produces one chunk per ━-delimited schedule, ensuring that financial
    tables (─-separated rows) are never split across chunk boundaries.
    The ``GENERAL CONDITIONS`` section, when present as an explicit section,
    is further subdivided at ``Clause N`` boundaries with per-clause metadata.
    """

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split a contract document into schedule- and clause-aware Chunk objects.

        Produces one chunk for the document header block, one chunk per
        schedule, and (when a ``GENERAL CONDITIONS`` section is present) one
        chunk per numbered clause within that section.

        Chunk content always begins with the schedule or clause heading so
        that each chunk is self-contained for embedding and citation display.

        Args:
            content: Full raw text of the contract file as read from disk.
            metadata: Must contain ``document_id``, ``document_type``, and
                ``source_path``.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects
            reflecting the document's schedule and clause structure.
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
                        "schedule_name": "Document Header",
                        "clause_id": "",
                        "has_table": False,
                        **header_fields,
                    },
                )
            )
            chunk_index += 1

        # --- One chunk per schedule, or per clause for GENERAL CONDITIONS -----
        i = 1
        while i + 1 < len(segments):
            section_title = segments[i].strip()
            section_body = segments[i + 1]
            i += 2

            if section_title == _GENERAL_CONDITIONS_TITLE:
                for clause_body, clause_id, clause_title in self._split_general_conditions(
                    section_body
                ):
                    if not clause_body.strip():
                        continue
                    heading = (
                        f"GENERAL CONDITIONS — Clause {clause_id} — {clause_title}"
                        if clause_id
                        else _GENERAL_CONDITIONS_TITLE
                    )
                    chunk_content = f"{heading}\n\n{clause_body.strip()}"
                    chunks.append(
                        self.build_chunk(
                            content=chunk_content,
                            document_id=document_id,
                            document_type=document_type,
                            source_path=source_path,
                            chunk_index=chunk_index,
                            extra_metadata={
                                "schedule_name": _GENERAL_CONDITIONS_TITLE,
                                "clause_id": clause_id,
                                "has_table": self._has_table(clause_body),
                            },
                        )
                    )
                    chunk_index += 1

            else:
                # Regular schedule — one chunk, table integrity guaranteed
                if not section_body.strip():
                    continue
                chunk_content = f"{section_title}\n\n{section_body.strip()}"
                chunks.append(
                    self.build_chunk(
                        content=chunk_content,
                        document_id=document_id,
                        document_type=document_type,
                        source_path=source_path,
                        chunk_index=chunk_index,
                        extra_metadata={
                            "schedule_name": section_title,
                            "clause_id": "",
                            "has_table": self._has_table(section_body),
                        },
                    )
                )
                chunk_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Private parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_on_separators(text: str) -> list[str]:
        """Split the full document text on ━-only separator lines.

        Iterates line by line and begins a new segment each time a ━ separator
        line is encountered.  The resulting list has the same interleaving
        structure as the SOP chunker:

        - ``segments[0]``  — document header block
        - ``segments[1]``, ``[3]``, ... — section title strings (odd indices)
        - ``segments[2]``, ``[4]``, ... — section body strings (even indices >= 2)

        The ─ table separator lines within section bodies are not affected by
        this split — only ━ lines trigger a new segment.

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
    def _split_general_conditions(
        body: str,
    ) -> list[tuple[str, str, str]]:
        """Split the GENERAL CONDITIONS body into one tuple per clause.

        Identifies ``Clause N — Title`` header lines and uses them as split
        boundaries.  Any preamble text before the first clause is prepended to
        the first clause chunk.  The execution block that follows the last
        clause (``EXECUTED as an Agreement:`` and signature lines) is left
        attached to that last clause's body so that signatory information is
        retained in context.

        If no ``Clause N —`` lines are found (e.g. GENERAL CONDITIONS is
        present but contains only free-form text), the entire body is returned
        as a single tuple with an empty clause_id.

        Args:
            body: The raw body text of the GENERAL CONDITIONS section
                (content between the ━ separator pair).

        Returns:
            List of ``(clause_body, clause_id, clause_title)`` tuples.
            ``clause_id`` is the clause number string (``"1"``, ``"2"``, …)
            or ``""`` for a fallback single-chunk case.
            ``clause_body`` does not include the ``Clause N — Title`` header
            line itself; the caller reconstructs the heading to avoid
            duplication.
        """
        lines = body.split("\n")

        # Locate all "Clause N — Title" lines
        clause_starts: list[tuple[int, str, str]] = []
        for idx, line in enumerate(lines):
            m = _CLAUSE_LINE.match(line.strip())
            if m:
                clause_starts.append((idx, m.group(1), m.group(2).strip()))

        if not clause_starts:
            return [(body, "", "")]

        # Preamble: lines before the first clause header
        preamble = "\n".join(lines[: clause_starts[0][0]]).strip()
        boundaries = [start for start, _, _ in clause_starts] + [len(lines)]

        result: list[tuple[str, str, str]] = []
        for slot, (start, clause_num, clause_title) in enumerate(clause_starts):
            end = boundaries[slot + 1]

            # Body begins on the line after the clause header
            clause_body = "\n".join(lines[start + 1 : end]).strip()

            # Attach preamble to the first clause only
            if slot == 0 and preamble:
                clause_body = preamble + "\n\n" + clause_body

            result.append((clause_body, clause_num, clause_title))

        return result

    @staticmethod
    def _has_table(text: str) -> bool:
        """Return True if the text contains at least one ─-only separator line.

        A ─ separator line (BOX DRAWINGS LIGHT HORIZONTAL, U+2500) is the
        marker used in the BuildCore corpus to delimit financial table rows,
        milestone schedules, and insurance requirement tables.  Its presence
        is a reliable signal that the chunk contains tabular data.

        Args:
            text: Section body or clause body text to inspect.

        Returns:
            ``True`` if any line in the text is composed solely of ─ characters
            (with optional surrounding whitespace), ``False`` otherwise.
        """
        for line in text.split("\n"):
            if _TABLE_SEPARATOR_LINE.match(line):
                return True
        return False

    @staticmethod
    def _parse_header_fields(header: str) -> dict[str, str]:
        """Parse key: value metadata fields from the document header block.

        Extracts fields such as Agreement No., Project, Client, Subcontractor,
        Execution Date, Commencement, and Practical Completion.  Multi-line
        values (e.g. the ABN and address lines that follow ``Client:`` or
        ``Subcontractor:``) are not captured — only the first line of each
        field is stored.

        Key normalisation: lowercased, dots removed, spaces and slashes
        replaced with underscores (e.g. ``"Agreement No."`` → ``"agreement_no"``,
        ``"Practical Completion"`` → ``"practical_completion"``).

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
