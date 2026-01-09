"""SEC document parser with table structure preservation."""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag
import html


@dataclass
class DocumentSection:
    """A section of a parsed document with metadata."""
    content: str
    section_type: str  # "text", "table", "header"
    section_name: str  # e.g., "Item 1A. Risk Factors"
    page_number: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """A fully parsed SEC document."""
    file_path: Path
    form_type: str
    filing_date: str
    company_name: str
    sections: List[DocumentSection]
    raw_text: str

    @property
    def total_pages(self) -> int:
        """Estimate total pages based on content."""
        # Rough estimate: ~3000 chars per page
        return max(1, len(self.raw_text) // 3000)


class DocumentParser:
    """Parse SEC HTML documents with table structure preservation."""

    # Common SEC section patterns
    SECTION_PATTERNS = {
        "10-K": [
            (r"(?:ITEM|Item)\s*1[.\s]*Business", "Item 1. Business"),
            (r"(?:ITEM|Item)\s*1A[.\s]*Risk\s*Factors", "Item 1A. Risk Factors"),
            (r"(?:ITEM|Item)\s*1B[.\s]*Unresolved\s*Staff", "Item 1B. Unresolved Staff Comments"),
            (r"(?:ITEM|Item)\s*2[.\s]*Properties", "Item 2. Properties"),
            (r"(?:ITEM|Item)\s*3[.\s]*Legal", "Item 3. Legal Proceedings"),
            (r"(?:ITEM|Item)\s*4[.\s]*Mine\s*Safety", "Item 4. Mine Safety"),
            (r"(?:ITEM|Item)\s*5[.\s]*Market", "Item 5. Market Information"),
            (r"(?:ITEM|Item)\s*6[.\s]*(?:Selected|Reserved)", "Item 6. Selected Financial Data"),
            (r"(?:ITEM|Item)\s*7[.\s]*Management", "Item 7. MD&A"),
            (r"(?:ITEM|Item)\s*7A[.\s]*Quantitative", "Item 7A. Market Risk"),
            (r"(?:ITEM|Item)\s*8[.\s]*Financial\s*Statements", "Item 8. Financial Statements"),
            (r"(?:ITEM|Item)\s*9[.\s]*Changes", "Item 9. Disagreements"),
            (r"(?:ITEM|Item)\s*9A[.\s]*Controls", "Item 9A. Controls and Procedures"),
            (r"(?:ITEM|Item)\s*10[.\s]*Directors", "Item 10. Directors and Officers"),
            (r"(?:ITEM|Item)\s*11[.\s]*Executive\s*Compensation", "Item 11. Executive Compensation"),
            (r"(?:ITEM|Item)\s*12[.\s]*Security\s*Ownership", "Item 12. Security Ownership"),
            (r"(?:ITEM|Item)\s*13[.\s]*Certain\s*Relationships", "Item 13. Related Party Transactions"),
            (r"(?:ITEM|Item)\s*14[.\s]*Principal\s*Account", "Item 14. Principal Accountant"),
            (r"(?:ITEM|Item)\s*15[.\s]*Exhibits", "Item 15. Exhibits"),
        ],
        "10-Q": [
            (r"(?:ITEM|Item)\s*1[.\s]*Financial\s*Statements", "Item 1. Financial Statements"),
            (r"(?:ITEM|Item)\s*2[.\s]*Management", "Item 2. MD&A"),
            (r"(?:ITEM|Item)\s*3[.\s]*Quantitative", "Item 3. Market Risk"),
            (r"(?:ITEM|Item)\s*4[.\s]*Controls", "Item 4. Controls and Procedures"),
            (r"(?:ITEM|Item)\s*1[.\s]*Legal", "Part II Item 1. Legal Proceedings"),
            (r"(?:ITEM|Item)\s*1A[.\s]*Risk\s*Factors", "Part II Item 1A. Risk Factors"),
            (r"(?:ITEM|Item)\s*6[.\s]*Exhibits", "Part II Item 6. Exhibits"),
        ],
        "8-K": [
            (r"(?:ITEM|Item)\s*1\.01", "Item 1.01 Entry into Material Agreement"),
            (r"(?:ITEM|Item)\s*1\.02", "Item 1.02 Termination of Material Agreement"),
            (r"(?:ITEM|Item)\s*2\.01", "Item 2.01 Acquisition/Disposition"),
            (r"(?:ITEM|Item)\s*2\.02", "Item 2.02 Results of Operations"),
            (r"(?:ITEM|Item)\s*2\.03", "Item 2.03 Direct Financial Obligation"),
            (r"(?:ITEM|Item)\s*5\.02", "Item 5.02 Director/Officer Changes"),
            (r"(?:ITEM|Item)\s*7\.01", "Item 7.01 Regulation FD Disclosure"),
            (r"(?:ITEM|Item)\s*8\.01", "Item 8.01 Other Events"),
        ],
    }

    def __init__(self):
        self.current_page = 1

    def parse(self, file_path: Path, form_type: str, filing_date: str) -> ParsedDocument:
        """
        Parse an SEC HTML document.

        Args:
            file_path: Path to the HTML file
            form_type: Type of SEC form (10-K, 10-Q, 8-K)
            filing_date: Filing date string

        Returns:
            ParsedDocument with extracted sections
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")

        # Extract company name
        company_name = self._extract_company_name(soup)

        # Remove scripts, styles, and hidden elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Parse sections
        sections = self._extract_sections(soup, form_type)

        # Get raw text for full-text search
        raw_text = soup.get_text(separator=" ", strip=True)
        raw_text = re.sub(r'\s+', ' ', raw_text)

        return ParsedDocument(
            file_path=file_path,
            form_type=form_type,
            filing_date=filing_date,
            company_name=company_name,
            sections=sections,
            raw_text=raw_text
        )

    def _extract_company_name(self, soup: BeautifulSoup) -> str:
        """Extract company name from document."""
        # Try common locations
        for selector in [
            'span[style*="font-weight:bold"]',
            'b', 'strong',
            '.companyName',
            '#companyName'
        ]:
            elements = soup.select(selector)
            for elem in elements[:5]:
                text = elem.get_text(strip=True)
                if len(text) > 5 and len(text) < 100:
                    if any(word in text.upper() for word in ["INC", "CORP", "LLC", "LTD", "CO"]):
                        return text

        return "Unknown Company"

    def _extract_sections(self, soup: BeautifulSoup, form_type: str) -> List[DocumentSection]:
        """Extract document sections with structure preservation."""
        sections = []
        current_section = "Preamble"
        current_content = []
        page_number = 1

        # Get section patterns for this form type
        patterns = self.SECTION_PATTERNS.get(form_type, [])

        # Process document body
        body = soup.find("body") or soup

        for element in body.descendants:
            if not isinstance(element, Tag):
                continue

            # Check for page breaks
            if self._is_page_break(element):
                page_number += 1
                continue

            # Check for section headers
            text = element.get_text(strip=True)
            new_section = self._detect_section(text, patterns)

            if new_section and new_section != current_section:
                # Save current section
                if current_content:
                    content = "\n".join(current_content)
                    if content.strip():
                        sections.append(DocumentSection(
                            content=content,
                            section_type="text",
                            section_name=current_section,
                            page_number=page_number
                        ))
                current_section = new_section
                current_content = []

            # Process tables specially
            if element.name == "table":
                table_md = self._table_to_markdown(element)
                if table_md:
                    sections.append(DocumentSection(
                        content=table_md,
                        section_type="table",
                        section_name=current_section,
                        page_number=page_number
                    ))
                continue

            # Skip if inside a table (already processed)
            if element.find_parent("table"):
                continue

            # Extract text content
            if element.name in ["p", "div", "span", "li", "h1", "h2", "h3", "h4", "h5", "h6"]:
                text = element.get_text(strip=True)
                if text and len(text) > 2:
                    # Avoid duplicates from nested elements
                    if not current_content or text != current_content[-1]:
                        current_content.append(text)

        # Save last section
        if current_content:
            content = "\n".join(current_content)
            if content.strip():
                sections.append(DocumentSection(
                    content=content,
                    section_type="text",
                    section_name=current_section,
                    page_number=page_number
                ))

        return sections

    def _detect_section(self, text: str, patterns: List[Tuple[str, str]]) -> Optional[str]:
        """Detect if text is a section header."""
        if not text or len(text) > 200:
            return None

        for pattern, section_name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return section_name

        return None

    def _is_page_break(self, element: Tag) -> bool:
        """Check if element represents a page break."""
        if element.name == "hr":
            return True
        style = element.get("style", "")
        if "page-break" in style:
            return True
        if element.get("class"):
            classes = " ".join(element.get("class", []))
            if "pagebreak" in classes.lower():
                return True
        return False

    def _table_to_markdown(self, table: Tag) -> str:
        """
        Convert HTML table to Markdown format for better LLM processing.

        Preserves structure while making it readable.
        """
        rows = []

        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                # Get cell text and clean it
                cell_text = td.get_text(strip=True)
                cell_text = re.sub(r'\s+', ' ', cell_text)

                # Handle colspan
                colspan = int(td.get("colspan", 1))
                cells.extend([cell_text] + [""] * (colspan - 1))

            if cells:
                rows.append(cells)

        if not rows:
            return ""

        # Normalize column count
        max_cols = max(len(row) for row in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # Build markdown table
        md_lines = []

        # Header row
        if rows:
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
            md_lines.append(header)
            md_lines.append(separator)

            # Data rows
            for row in rows[1:]:
                md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)
