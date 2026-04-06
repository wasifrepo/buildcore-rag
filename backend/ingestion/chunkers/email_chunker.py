"""Email thread chunker for BuildCore incident email documents.

Incident email threads in the BuildCore corpus are multi-message files where
individual messages are delimited by ━ separator lines.  Each message opens
with a compact header block (From, To, optional CC, Date, Subject) followed
by a blank line and the message body.

Chunking strategy
-----------------
Each message in the thread becomes exactly one chunk — messages are never
merged and a single message is never split.  The full message text (header
block + body) is stored as the chunk content so that sender identity, date,
and subject are present in the embedding and can be surfaced verbatim in
citations.

The ━ separator lines are purely delimiters; unlike SOPs and contracts they
do not frame a title.  Splitting on ━ lines therefore yields one raw segment
per message directly (no alternating title/body pairs).

Metadata per chunk
------------------
``sender``        — display name of the From address (e.g. "Tom Nguyen")
``sender_email``  — email address of the sender (e.g. "t.nguyen@buildcore.com.au")
``date``          — raw date string as it appears in the header
``subject``       — raw subject line (including any "RE:" prefix)
``recipients``    — combined To + CC as a list of "Name <email>" strings
``has_cc``        — True when a CC field is present in the message header
``message_index`` — 0-based position of this message within the thread
"""

import re

from generation.schemas import Chunk
from ingestion.chunkers.base import BaseChunker

# ---------------------------------------------------------------------------
# Module-level compiled patterns
# ---------------------------------------------------------------------------

# Section delimiter: line of ━ characters (U+2501 heavy horizontal)
_SEPARATOR_LINE: re.Pattern[str] = re.compile(r"^━+$")

# "Name <email>" pattern used to extract sender display name and address
# from the From field value (after the "From: " prefix has been removed).
_NAME_EMAIL: re.Pattern[str] = re.compile(r"^(.+?)\s+<([^>]+)>$")


class EmailChunker(BaseChunker):
    """Chunker for BuildCore incident email thread documents.

    Splits a multi-message thread file into one :class:`~generation.schemas.Chunk`
    per message, preserving the full header block in the chunk content so that
    sender, date, and subject are always co-embedded with the message body.
    """

    def chunk(self, content: str, metadata: dict) -> list[Chunk]:
        """Split an incident email thread into one Chunk per message.

        Messages are identified by splitting on ━ separator lines.  Each
        resulting segment is expected to begin (after stripping) with a
        ``From:`` header line.  Segments that are empty after stripping are
        skipped silently.

        Args:
            content: Full raw text of the email thread file as read from disk.
            metadata: Must contain ``document_id``, ``document_type``, and
                ``source_path``.

        Returns:
            Ordered list of :class:`~generation.schemas.Chunk` objects, one
            per message, in thread chronological order (message_index 0 first).
        """
        document_id: str = metadata["document_id"]
        document_type: str = metadata["document_type"]
        source_path: str = metadata["source_path"]

        text = self.clean_whitespace(content)
        segments = self._split_on_separators(text)

        chunks: list[Chunk] = []
        message_index = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            header_fields = self._parse_message_header(segment)

            chunks.append(
                self.build_chunk(
                    content=segment,
                    document_id=document_id,
                    document_type=document_type,
                    source_path=source_path,
                    chunk_index=message_index,
                    extra_metadata={
                        "message_index": message_index,
                        "sender": header_fields["sender"],
                        "sender_email": header_fields["sender_email"],
                        "date": header_fields["date"],
                        "subject": header_fields["subject"],
                        "recipients": header_fields["recipients"],
                        "has_cc": header_fields["has_cc"],
                    },
                )
            )
            message_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_on_separators(text: str) -> list[str]:
        """Split the thread text on ━-only lines to yield one segment per message.

        Unlike the SOP and contract chunkers where ━ lines frame a section
        title, here they are purely delimiters between consecutive messages.
        Each resulting segment contains the full text of one message (header
        block and body) and requires no further pairing logic.

        Leading and trailing blank lines within a segment are preserved and
        stripped by the caller.

        Args:
            text: Whitespace-normalised full thread text.

        Returns:
            List of raw message segments in thread order.
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

    @classmethod
    def _parse_message_header(cls, segment: str) -> dict:
        """Extract structured fields from the header block of a message segment.

        Reads lines from the top of the segment until a blank line is reached
        (the header/body separator).  Recognises ``From:``, ``To:``, ``CC:``,
        ``Date:``, and ``Subject:`` fields.

        All header field values are extracted with ``split(':', 1)`` so that
        colons inside values (e.g. ``11:06 AM`` in Date, or ``RE:`` in Subject)
        are preserved correctly.

        ``recipients`` is assembled by combining the To and CC address lists.
        ``has_cc`` is set to ``True`` only when a ``CC:`` line is present in
        the header; its absence (as in the first message of INC-2024-009) is
        represented as ``False``.

        Args:
            segment: A single stripped message segment beginning with
                ``From: Name <email>``.

        Returns:
            Dict with keys: ``sender``, ``sender_email``, ``date``,
            ``subject``, ``recipients`` (list of str), ``has_cc`` (bool).
            Any field not found in the header defaults to an empty string or
            empty list.
        """
        result: dict = {
            "sender": "",
            "sender_email": "",
            "date": "",
            "subject": "",
            "recipients": [],
            "has_cc": False,
        }

        to_raw: str = ""
        cc_raw: str = ""

        for line in segment.split("\n"):
            if not line.strip():
                # Blank line marks the end of the header block
                break

            if line.startswith("From:"):
                value = line.split(":", 1)[1].strip()
                m = _NAME_EMAIL.match(value)
                if m:
                    result["sender"] = m.group(1).strip()
                    result["sender_email"] = m.group(2).strip()

            elif line.startswith("To:"):
                to_raw = line.split(":", 1)[1].strip()

            elif line.startswith("CC:"):
                cc_raw = line.split(":", 1)[1].strip()
                result["has_cc"] = True

            elif line.startswith("Date:"):
                result["date"] = line.split(":", 1)[1].strip()

            elif line.startswith("Subject:"):
                result["subject"] = line.split(":", 1)[1].strip()

        recipients: list[str] = []
        if to_raw:
            recipients.extend(cls._parse_address_list(to_raw))
        if cc_raw:
            recipients.extend(cls._parse_address_list(cc_raw))
        result["recipients"] = recipients

        return result

    @staticmethod
    def _parse_address_list(raw: str) -> list[str]:
        """Split a semicolon-separated address field into individual entries.

        Handles both single-address fields (``"Name <email>"``) and
        multi-address fields (``"Name <email>; Name <email>"``), as used
        in the To and CC lines of the BuildCore corpus.

        Args:
            raw: The raw value of a To or CC header field, after the field
                label and colon have been removed.

        Returns:
            List of stripped ``"Name <email>"`` strings.  Empty strings
            are excluded.
        """
        return [entry.strip() for entry in raw.split(";") if entry.strip()]
