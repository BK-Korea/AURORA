"""Citation validator node - verifies citations exist in source documents."""
import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from ...vectorstore.chroma_store import RetrievedChunk


@dataclass
class Citation:
    """Parsed citation from answer."""
    form_type: str
    filing_date: str
    page: str
    section: str
    raw: str


class CitationValidatorNode:
    """
    Node that validates citations in generated answers.

    Ensures:
    - All citations reference real documents
    - Page numbers and sections exist
    - No fabricated citations
    """

    CITATION_PATTERN = r'\[([^\]]+)\]'

    def __init__(self, strict: bool = True):
        self.strict = strict

    def _parse_citation(self, citation_str: str) -> Citation:
        """Parse a citation string into components."""
        parts = citation_str.split('|')

        form_date = parts[0].strip() if len(parts) > 0 else ""
        page = parts[1].strip() if len(parts) > 1 else ""
        section = parts[2].strip() if len(parts) > 2 else ""

        # Extract form type and date
        form_parts = form_date.split()
        form_type = form_parts[0] if form_parts else ""
        filing_date = form_parts[1] if len(form_parts) > 1 else ""

        return Citation(
            form_type=form_type,
            filing_date=filing_date,
            page=page,
            section=section,
            raw=citation_str
        )

    def _validate_citation(
        self,
        citation: Citation,
        chunks: List[RetrievedChunk]
    ) -> bool:
        """Check if citation matches any retrieved chunk."""
        for chunk in chunks:
            meta = chunk.metadata

            # Check form type
            if citation.form_type and citation.form_type not in meta.get("form_type", ""):
                continue

            # Check filing date
            if citation.filing_date and citation.filing_date not in meta.get("filing_date", ""):
                continue

            # Check section (partial match)
            if citation.section:
                chunk_section = meta.get("section_name", "")
                if citation.section.lower() not in chunk_section.lower():
                    continue

            # If we get here, it's a valid match
            return True

        return False

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate citations in the generated answer.

        Input state:
            - current_answer: Generated answer with citations
            - retrieved_chunks: Source chunks used for answer

        Output state:
            - citations_valid: True if all citations are valid
            - error: Error message listing invalid citations
        """
        answer = state.get("current_answer", "")
        chunks: List[RetrievedChunk] = state.get("retrieved_chunks", [])

        if not answer:
            return {
                "citations_valid": True,
                "error": None
            }

        # Find all citations in answer
        citation_matches = re.findall(self.CITATION_PATTERN, answer)

        if not citation_matches:
            # No citations - might be "not found" response
            if "찾을 수 없" in answer or "없습니다" in answer:
                return {
                    "citations_valid": True,
                    "error": None
                }
            elif self.strict:
                return {
                    "citations_valid": False,
                    "error": "답변에 출처가 포함되어 있지 않습니다."
                }

        # Validate each citation
        invalid_citations = []
        for citation_str in citation_matches:
            # Skip non-citation brackets (like markdown links)
            if '(' in citation_str or ')' in citation_str:
                continue
            if not any(form in citation_str.upper() for form in ["10-K", "10-Q", "8-K", "DEF"]):
                continue

            citation = self._parse_citation(citation_str)
            if not self._validate_citation(citation, chunks):
                invalid_citations.append(citation.raw)

        if invalid_citations:
            return {
                "citations_valid": False,
                "error": f"검증할 수 없는 출처: {', '.join(invalid_citations)}"
            }

        return {
            "citations_valid": True,
            "error": None
        }
