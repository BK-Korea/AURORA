"""Document chunking with metadata preservation."""
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .parser import ParsedDocument, DocumentSection


@dataclass
class DocumentChunk:
    """A chunk of document with full provenance metadata."""
    content: str
    metadata: Dict

    @property
    def citation(self) -> str:
        """Generate citation string for this chunk."""
        return (
            f"[{self.metadata.get('form_type', 'Unknown')} "
            f"{self.metadata.get('filing_date', '')} | "
            f"Page {self.metadata.get('page_number', '?')} | "
            f"{self.metadata.get('section_name', 'Unknown Section')}]"
        )


class DocumentChunker:
    """
    Chunk documents while preserving metadata for citation.

    Each chunk maintains:
    - Source document (form type, filing date)
    - Page number
    - Section name
    - Chunk type (text/table)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Chunk a parsed document into retrieval-ready chunks.

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of DocumentChunk with full metadata
        """
        chunks = []
        base_metadata = {
            "file_path": str(doc.file_path),
            "form_type": doc.form_type,
            "filing_date": doc.filing_date,
            "company_name": doc.company_name,
        }

        for section in doc.sections:
            section_metadata = {
                **base_metadata,
                "section_name": section.section_name,
                "section_type": section.section_type,
                "page_number": section.page_number or 1,
            }

            if section.section_type == "table":
                # Keep tables as single chunks if possible
                if len(section.content) <= self.chunk_size * 1.5:
                    chunks.append(DocumentChunk(
                        content=section.content,
                        metadata={**section_metadata, "is_table": True}
                    ))
                else:
                    # Split large tables by rows
                    table_chunks = self._split_table(section.content)
                    for i, chunk_content in enumerate(table_chunks):
                        chunks.append(DocumentChunk(
                            content=chunk_content,
                            metadata={
                                **section_metadata,
                                "is_table": True,
                                "table_part": i + 1
                            }
                        ))
            else:
                # Split text content
                text_chunks = self.text_splitter.split_text(section.content)
                for i, chunk_content in enumerate(text_chunks):
                    if len(chunk_content) >= self.min_chunk_size:
                        chunks.append(DocumentChunk(
                            content=chunk_content,
                            metadata={
                                **section_metadata,
                                "chunk_index": i,
                                "is_table": False
                            }
                        ))

        return chunks

    def _split_table(self, table_content: str) -> List[str]:
        """Split a large table while preserving header."""
        lines = table_content.split("\n")
        if len(lines) <= 3:
            return [table_content]

        # Keep header (first two lines: header row + separator)
        header = "\n".join(lines[:2])
        data_lines = lines[2:]

        chunks = []
        current_chunk_lines = []
        current_length = len(header)

        for line in data_lines:
            if current_length + len(line) + 1 > self.chunk_size:
                if current_chunk_lines:
                    chunk = header + "\n" + "\n".join(current_chunk_lines)
                    chunks.append(chunk)
                current_chunk_lines = [line]
                current_length = len(header) + len(line)
            else:
                current_chunk_lines.append(line)
                current_length += len(line) + 1

        # Last chunk
        if current_chunk_lines:
            chunk = header + "\n" + "\n".join(current_chunk_lines)
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, docs: List[ParsedDocument]) -> List[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
