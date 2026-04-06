"""SOP document chunker for BuildCore safety Standard Operating Procedures.

SOPs in the BuildCore corpus are structured with ━ separator lines that frame
numbered section headers (e.g. ``1. PURPOSE``, ``6. FALL PROTECTION
REQUIREMENTS BY WORK TYPE``).  Subsections use dotted numbering: ``5.1``,
``6.2``, ``6.2.1``, etc.

Chunking strategy
-----------------
1. The document header block (before the first ━ line) becomes a single chunk
   carrying the parsed key-value fields (Document ID, Title, Version, etc.)
   as metadata.
2. Each numbered section becomes one chunk.  If a section exceeds
   ``MAX_SECTION_CHARS`` characters it is subdivided at its top-level N.M
   subsection boundaries.  Deep subsections (N.M.P and below) are never used
   as split points, so they always remain with their parent N.M subsection.
3. The unnumbered ``DOCUMENT REVISION HISTORY`` section is always a single
   chunk.

Every chunk carries ``section_number``, ``section_title``, and ``subsections``
(a list of N.M / N.M.P labels found in the chunk body) in its metadata.
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Module-level compiled patterns
# ---------------------------------------------------------------------------

# A line consisting entirely of ━ characters — section delimiter
_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^━+$")

# Numbered section header: "1. PURPOSE", "10. RECORDKEEPING AND REPORTING"
_NUMBERED_HEADER: re.Pattern[str] = re.compile(r"^(\d+)\.\s+(.+)$")

# Subsection reference line: "5.1 Site Safety Manager",
# "  6.2.1 Guardrail System (preferred)"
_SUBSECTION_LINE: re.Pattern[str] = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(.+)$"
)

# Key-value field in document header: "Effective Date: 01 March 2024",
# "Approved By:    Director of Operations"
_HEADER_FIELD: re.Pattern[str] = re.compile(
    r"^([A-Za-z][A-Za-z /]+):\s+(.+)$"
)


class SOPChunker(BaseChunker):
    """Chunker for BuildCore safety Standard Operating Procedures.

    Parses the ━-delimited section structure of SOP files and produces
    one chunk per top-level section (or several if a section is long),
    plus a leading chunk for the document header block.

    Attributes:
        MAX_SECTION_CHARS: Character threshold above which a section body
            is subdivided at its top-level subsection (N.M) boundaries.
            All sections in the current corpus fall below this threshold,
            so each section produces exactly one chunk; the split logic is
            present for correctness with larger future documents.
    """

    MAX_SECTION_CHARS: int = 3000

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split a safety SOP into section-aware Chunk objects.

        Produces one chunk for the document header block (section_number ``""``,
        section_title ``"Document Header"``), then one or more chunks per
        numbered or unnumbered section.  Chunk content always begins with the
        section heading so each chunk is self-contained for embedding.

        Args:
            content: Full raw text of the SOP file as read from disk.
            metadata: Must contain ``document_id``, ``document_type``, and
                ``source_path``.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects
            reflecting the document's section structure.
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
                        "subsections": [],
                        **header_fields,
                    },
                )
            )
            chunk_index += 1

        # --- One or more chunks per section ----------------------------------
        i = 1
        while i + 1 < len(segments):
            section_number, section_title = self._parse_section_header(
                segments[i].strip()
            )
            section_body = segments[i + 1]
            i += 2

            for part_body, part_number, part_title in self._split_section(
                section_body, section_number, section_title
            ):
                if not part_body.strip():
                    continue

                # Prepend the heading so each chunk is self-contained
                heading = (
                    f"{part_number}. {part_title}"
                    if part_number
                    else part_title
                )
                chunk_content = f"{heading}\n\n{part_body.strip()}"
                subsections = self._extract_subsection_labels(part_body)

                chunks.append(
                    self.build_chunk(
                        content=chunk_content,
                        document_id=document_id,
                        document_type=document_type,
                        source_path=source_path,
                        chunk_index=chunk_index,
                        extra_metadata={
                            "section_number": part_number,
                            "section_title": part_title,
                            "subsections": subsections,
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

        Iterates line by line.  Each time a separator line is encountered a
        new segment begins.  This gives:

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

        Handles both numbered sections (``"1. PURPOSE"`` -> ``("1", "PURPOSE")``)
        and unnumbered sections (``"DOCUMENT REVISION HISTORY"`` ->
        ``("", "DOCUMENT REVISION HISTORY")``).

        Args:
            raw: Stripped section title string extracted from between ━ lines.

        Returns:
            Two-tuple of ``(section_number, section_title)``.
            ``section_number`` is an empty string for unnumbered sections.
        """
        m = _NUMBERED_HEADER.match(raw)
        if m:
            return m.group(1), m.group(2).strip()
        return "", raw

    def _split_section(
        self,
        body: str,
        section_number: str,
        section_title: str,
    ) -> list[tuple[str, str, str]]:
        """Split a section body into sub-chunks if it exceeds MAX_SECTION_CHARS.

        Sections within the character limit are returned as a single-element
        list.  Oversized sections are subdivided at their top-level N.M
        subsection boundaries only — deep subsections (N.M.P and below) are
        never used as split points, ensuring they always stay with their parent
        N.M block.

        Any preamble text (content before the first subsection line) is
        attached to the first sub-chunk so no content is lost.

        The subsection header line itself is excluded from ``part_body`` in the
        returned tuples; the caller reconstructs the heading from
        ``part_number`` and ``part_title`` to avoid duplication when the
        heading is prepended to the chunk content.

        Args:
            body: Raw section body text (content between the ━ separator pair).
            section_number: Top-level section number string, e.g. ``"6"``.
            section_title: Section title string, e.g.
                ``"FALL PROTECTION REQUIREMENTS BY WORK TYPE"``.

        Returns:
            List of ``(part_body, part_number, part_title)`` tuples, one per
            output chunk.  For unsplit sections the tuple carries the original
            ``section_number`` and ``section_title``.  For sub-chunks,
            ``part_number`` is the N.M subsection number and ``part_title``
            is ``"{section_title} -- {subsection_title}"``.
        """
        if len(body) <= self.MAX_SECTION_CHARS:
            return [(body, section_number, section_title)]

        lines = body.split("\n")

        # Find indices of top-level subsection lines (N.M only, not N.M.P)
        split_points: list[int] = []
        for idx, line in enumerate(lines):
            m = _SUBSECTION_LINE.match(line)
            if m and m.group(1).count(".") == 1:
                split_points.append(idx)

        if not split_points:
            # No subsection boundaries found — cannot subdivide safely
            return [(body, section_number, section_title)]

        # Preamble: any lines before the first subsection header
        preamble = "\n".join(lines[: split_points[0]]).strip()
        boundaries = split_points + [len(lines)]

        result: list[tuple[str, str, str]] = []
        for slot, start in enumerate(split_points):
            end = boundaries[slot + 1]

            # Body lines start after the subsection header line
            sub_body = "\n".join(lines[start + 1 : end]).strip()

            # Preamble (intro text before first subsection) joins the first chunk
            if slot == 0 and preamble:
                sub_body = preamble + "\n\n" + sub_body

            m = _SUBSECTION_LINE.match(lines[start])
            sub_number = m.group(1) if m else ""
            sub_title_raw = m.group(2).strip() if m else lines[start].strip()
            combined_title = f"{section_title} -- {sub_title_raw}"

            result.append((sub_body, sub_number, combined_title))

        return result

    @staticmethod
    def _extract_subsection_labels(text: str) -> list[str]:
        """Extract all subsection reference labels present in a text block.

        Scans every line for the N.M or N.M.P subsection pattern and returns
        matched labels in document order.  Used to populate the ``subsections``
        metadata field, which enables faceted filtering during retrieval.

        Args:
            text: Section or sub-chunk body text (the heading line is excluded
                by the caller and reconstructed separately to avoid counting
                the split-point itself as a subsection reference).

        Returns:
            Ordered list of subsection label strings, e.g.
            ``["5.1 Site Safety Manager", "5.2 Supervisors and Foremen"]``.
            Empty list if no subsection lines are present.
        """
        labels: list[str] = []
        for line in text.split("\n"):
            m = _SUBSECTION_LINE.match(line)
            if m:
                labels.append(f"{m.group(1)} {m.group(2).strip()}")
        return labels

    @staticmethod
    def _parse_header_fields(header: str) -> dict[str, str]:
        """Parse key: value metadata fields from the document header block.

        Extracts structured fields such as Document ID, Title, Version, and
        Effective Date.  Keys are lowercased with spaces and slashes replaced
        by underscores for consistent downstream access.

        Only lines matching the ``KEY:  value`` pattern are extracted; free-text
        lines like ``"BUILDCORE OPERATIONS"`` and ``"STANDARD OPERATING
        PROCEDURE"`` are ignored.

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
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                fields[key] = m.group(2).strip()
        return fields
