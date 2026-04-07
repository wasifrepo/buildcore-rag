"""Document type classifier for the BuildCore ingestion pipeline.

Determines which of the five corpus document types a file belongs to so that
the appropriate chunker can be selected downstream.

Classification strategy (in order of precedence):
1. Parent directory name — reliable when files live under the canonical
   ``data/raw/<type>/`` folder structure.
2. Filename prefix pattern — fallback for files that have been moved or are
   passed from an arbitrary path, using the naming conventions observed across
   the entire corpus.

Raises ``ValueError`` for any file that cannot be mapped to a known type.
"""

import re
from pathlib import Path

from generation.schemas import DocumentType


# ---------------------------------------------------------------------------
# Folder-name → DocumentType mapping
# Matches any component of the file path, so works for both absolute and
# relative paths regardless of how deeply the type folder is nested.
# ---------------------------------------------------------------------------
_FOLDER_MAP: dict[str, DocumentType] = {
    "safety_sops": DocumentType.SAFETY_SOP,
    "contracts": DocumentType.CONTRACT,
    "incident_emails": DocumentType.INCIDENT_EMAIL,
    "maintenance_manuals": DocumentType.MAINTENANCE_MANUAL,
    "compliance_checklists": DocumentType.COMPLIANCE_CHECKLIST,
    "regulatory_docs": DocumentType.REGULATORY_DOC,
}

# ---------------------------------------------------------------------------
# Filename-prefix patterns → DocumentType mapping
# Ordered so that more-specific patterns (e.g. SC-PMCL-) are evaluated before
# the broader SC-\d{4}- contract pattern, which shares the "SC-" prefix.
# ---------------------------------------------------------------------------
_FILENAME_PATTERNS: list[tuple[re.Pattern[str], DocumentType]] = [
    (re.compile(r"^SOP-\d+", re.IGNORECASE), DocumentType.SAFETY_SOP),
    (re.compile(r"^INC-\d{4}-\d{3}", re.IGNORECASE), DocumentType.INCIDENT_EMAIL),
    (re.compile(r"^MAINT-", re.IGNORECASE), DocumentType.MAINTENANCE_MANUAL),
    (re.compile(r"^(SSIC|SC-PMCL)-", re.IGNORECASE), DocumentType.COMPLIANCE_CHECKLIST),
    (re.compile(r"^SC-\d{4}-\d{3}", re.IGNORECASE), DocumentType.CONTRACT),
    (re.compile(r"^OSHA\d+", re.IGNORECASE), DocumentType.REGULATORY_DOC),
]


def classify_document(file_path: str | Path) -> DocumentType:
    """Classify a document file into one of the five BuildCore document types.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        The ``DocumentType`` enum value that corresponds to this file.

    Raises:
        ValueError: If the document type cannot be determined from either the
            directory name or the filename prefix.
    """
    path = Path(file_path)

    # --- Primary: directory name -------------------------------------------
    # Walk every component of the path so the function works regardless of
    # whether the caller passes an absolute path, a relative path, or just the
    # filename.
    for part in path.parts:
        if part in _FOLDER_MAP:
            return _FOLDER_MAP[part]

    # --- Fallback: filename prefix pattern ----------------------------------
    stem = path.stem  # filename without extension
    for pattern, doc_type in _FILENAME_PATTERNS:
        if pattern.match(stem):
            return doc_type

    raise ValueError(
        f"Cannot determine document type for '{file_path}'. "
        "Ensure the file is located under a recognised corpus directory "
        "(safety_sops, contracts, incident_emails, maintenance_manuals, "
        "compliance_checklists, regulatory_docs) or has a recognised filename prefix."
    )
