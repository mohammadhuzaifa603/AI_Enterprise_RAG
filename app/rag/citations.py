"""
Citation construction & grounding.

Builds the evidence block handed to the LLM and, critically, validates
any citation the LLM returns against the actual retrieved chunk metadata.
An LLM (real or mock) is never trusted to invent a
(document, page, chunk_id) triple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.models import Chunk


@dataclass
class EvidenceItem:
    chunk_id: str
    document_name: str
    page: int
    text: str
    chunk_type: str = "text"


def build_evidence_items(
    chunks_with_docnames: List[tuple[Chunk, str]],
) -> List[EvidenceItem]:

    return [
        EvidenceItem(
            chunk_id=c.id,
            document_name=doc_name,
            page=c.page_number,
            text=c.text,
            chunk_type=getattr(
                c,
                "chunk_type",
                "text",
            ),
        )
        for c, doc_name in chunks_with_docnames
    ]


def format_evidence_block(
    items: List[EvidenceItem],
) -> str:

    blocks = []

    for i, item in enumerate(
        items,
        start=1,
    ):

        blocks.append(
            f"[EVIDENCE {i}"
            f"|doc={item.document_name}"
            f"|page={item.page}"
            f"|type={item.chunk_type}"
            f"|chunk={item.chunk_id}]\n"
            f"{item.text}"
        )

    return "\n\n".join(blocks)


def validate_citation(
    citation: Dict,
    evidence_items: List[EvidenceItem],
) -> Optional[EvidenceItem]:
    """Return the matching EvidenceItem if grounded."""

    chunk_id = citation.get("chunk_id")

    for item in evidence_items:

        if item.chunk_id == chunk_id:
            return item

    # Fall back to document + page.
    doc = citation.get("document")
    page = citation.get("page")

    for item in evidence_items:

        if (
            item.document_name == doc
            and item.page == page
        ):
            return item

    return None


def ground_citations(
    raw_citations: List[Dict],
    evidence_items: List[EvidenceItem],
) -> List[Dict]:
    """Validate LLM citations against retrieved evidence."""

    grounded = []

    for citation in raw_citations:

        match = validate_citation(
            citation,
            evidence_items,
        )

        if match is None:
            continue

        grounded.append(
            {
                "document": match.document_name,
                "page": match.page,
                "chunk_id": match.chunk_id,
                "chunk_type": match.chunk_type,
                "excerpt": match.text[:400],
                "claim_text": citation.get(
                    "claim_text",
                    "",
                ),
            }
        )

    return grounded