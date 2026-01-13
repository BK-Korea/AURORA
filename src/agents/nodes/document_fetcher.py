"""Document fetcher node - downloads SEC filings."""
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from ...document.sec_downloader import SECDownloader, SECFiling, CompanyInfo
from ...document.parser import DocumentParser, ParsedDocument
from ...document.chunker import DocumentChunker, DocumentChunk
from ...vectorstore.chroma_store import ChromaStore


class DocumentFetcherNode:
    """
    Node that downloads and processes SEC filings.

    Handles:
    - Downloading filings from SEC EDGAR
    - Parsing HTML documents
    - Chunking for vector storage
    - Indexing in vector store
    """

    def __init__(
        self,
        downloader: SECDownloader,
        parser: DocumentParser,
        chunker: DocumentChunker,
        vector_store: ChromaStore,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.downloader = downloader
        self.parser = parser
        self.chunker = chunker
        self.vector_store = vector_store
        self.progress_callback = progress_callback

    def _report_progress(self, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download and process SEC filings for a company.

        Input state:
            - company_info: CompanyInfo object
            - years_requested: Number of years to download

        Output state:
            - filings: List of downloaded SECFiling
            - documents_processed: True if successful
            - chunks_indexed: Number of chunks added to vector store
            - error: Error message if failed
        """
        company_info: CompanyInfo = state.get("company_info")
        years = state.get("years_requested", 3)

        if not company_info:
            return {
                "filings": [],
                "documents_processed": False,
                "chunks_indexed": 0,
                "error": "회사 정보가 없습니다."
            }

        self._report_progress(f"Downloading SEC filings for {company_info.name}...")

        # Download filings (including foreign company forms: 20-F, 6-K)
        filings = self.downloader.download_filings(
            cik=company_info.cik,
            form_types=["10-K", "10-Q", "8-K", "20-F", "6-K"],
            years=years,
            progress_callback=self._report_progress
        )

        if not filings:
            return {
                "filings": [],
                "documents_processed": False,
                "chunks_indexed": 0,
                "error": "다운로드된 문서가 없습니다."
            }

        self._report_progress(f"Downloaded {len(filings)} filings. Parsing...")

        # Parse documents
        parsed_docs = []
        for filing in filings:
            try:
                parsed = self.parser.parse(
                    file_path=filing.file_path,
                    form_type=filing.form_type,
                    filing_date=filing.filing_date
                )
                # Add ticker information to parsed document if available
                if company_info.ticker:
                    # Store ticker in a way that chunker can access
                    parsed.ticker = company_info.ticker
                parsed_docs.append(parsed)
            except Exception as e:
                self._report_progress(f"Warning: Failed to parse {filing.file_path}: {e}")

        self._report_progress(f"Parsed {len(parsed_docs)} documents. Chunking...")

        # Chunk documents with company info (including ticker)
        all_chunks = self.chunker.chunk_documents(parsed_docs, company_ticker=company_info.ticker)

        self._report_progress(f"Created {len(all_chunks)} chunks. Indexing...")

        # Index in vector store
        chunks_added = self.vector_store.add_chunks(
            chunks=all_chunks,
            progress_callback=self._report_progress
        )

        self._report_progress(f"Indexed {chunks_added} chunks successfully.")

        return {
            "filings": filings,
            "documents_processed": True,
            "chunks_indexed": chunks_added,
            "error": None
        }
