"""Chroma vector store with metadata filtering for citations."""
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from ..document.chunker import DocumentChunk


@dataclass
class RetrievedChunk:
    """A retrieved chunk with similarity score."""
    content: str
    metadata: Dict
    score: float

    @property
    def citation(self) -> str:
        """Generate citation string."""
        return (
            f"[{self.metadata.get('form_type', 'Unknown')} "
            f"{self.metadata.get('filing_date', '')} | "
            f"Page {self.metadata.get('page_number', '?')} | "
            f"{self.metadata.get('section_name', 'Unknown Section')}]"
        )

    def to_context_string(self) -> str:
        """Format chunk for LLM context."""
        return f"{self.citation}\n{self.content}"


class ChromaStore:
    """
    Chroma vector store wrapper with citation support.

    Features:
    - Persistent storage
    - Metadata filtering (by form type, date, section)
    - Retrieval with citations
    """

    def __init__(
        self,
        persist_dir: Path,
        embeddings: Embeddings,
        collection_name: str = "sec_documents"
    ):
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        self.collection_name = collection_name

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk to add
            batch_size: Batch size for embedding
            progress_callback: Optional progress callback

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Prepare data
            texts = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]
            ids = [f"{c.metadata.get('file_path', 'unknown')}_{i+j}" for j, c in enumerate(batch)]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            total_added += len(batch)

            if progress_callback:
                progress_callback(f"Indexed {total_added}/{len(chunks)} chunks")

        return total_added

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_form_types: Optional[List[str]] = None,
        filter_sections: Optional[List[str]] = None,
        filter_company_name: Optional[str] = None,
        filter_min_date: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_form_types: Optional list of form types to filter
            filter_sections: Optional list of sections to filter
            filter_company_name: Optional company name to filter (exact match)
            filter_min_date: Optional minimum filing date (YYYY-MM-DD format)
            min_score: Minimum similarity score

        Returns:
            List of RetrievedChunk with scores
        """
        # Build where clause for filtering
        where = None
        conditions = []
        if filter_form_types:
            conditions.append({"form_type": {"$in": filter_form_types}})
        if filter_sections:
            conditions.append({"section_name": {"$in": filter_sections}})
        if filter_company_name:
            conditions.append({"company_name": filter_company_name})
        if filter_min_date:
            # Filter by filing_date >= filter_min_date
            conditions.append({"filing_date": {"$gte": filter_min_date}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievedChunk
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                # Convert distance to similarity score (cosine)
                score = 1 - distance

                if score >= min_score:
                    retrieved.append(RetrievedChunk(
                        content=doc,
                        metadata=meta,
                        score=score
                    ))

        return retrieved

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection with pagination to avoid OOM."""
        count = self.collection.count()

        company_names = set()
        company_tickers = {}
        filing_dates = []

        if count == 0:
            return {
                "total_chunks": 0,
                "collection_name": self.collection_name,
                "persist_dir": str(self.persist_dir),
                "indexed_companies": [],
                "indexed_tickers": {},
                "latest_filing_date": None,
                "earliest_filing_date": None,
            }

        # Paginated metadata retrieval to prevent memory overflow
        batch_size = 5000
        offset = 0
        while offset < count:
            try:
                batch = self.collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
            except Exception:
                break

            metadatas = batch.get("metadatas", [])
            if not metadatas:
                break

            for meta in metadatas:
                if not meta:
                    continue
                if "company_name" in meta:
                    cname = meta["company_name"]
                    company_names.add(cname)
                    if "company_ticker" in meta and meta["company_ticker"]:
                        if cname not in company_tickers:
                            company_tickers[cname] = meta["company_ticker"]
                if "filing_date" in meta:
                    filing_dates.append(meta["filing_date"])

            offset += len(metadatas)

            # Early exit: once we have all companies and dates, no need to scan more
            if len(company_names) > 0 and offset >= min(count, 20000):
                break

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "indexed_companies": list(company_names),
            "indexed_tickers": company_tickers,
            "latest_filing_date": max(filing_dates) if filing_dates else None,
            "earliest_filing_date": min(filing_dates) if filing_dates else None,
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_by_company(self, company_name: str) -> int:
        """Delete all chunks for a specific company."""
        # Get IDs to delete
        results = self.collection.get(
            where={"company_name": company_name},
            include=[]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0
