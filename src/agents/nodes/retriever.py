"""Retriever node - searches vector store for relevant chunks."""
from typing import Dict, Any, List, Optional

from ...vectorstore.chroma_store import ChromaStore, RetrievedChunk


class RetrieverNode:
    """
    Node that retrieves relevant document chunks for a query.

    Features:
    - Semantic search using embeddings
    - Metadata filtering (form type, section)
    - Relevance scoring
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        top_k: int = 10,
        min_score: float = 0.3
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for the current query.

        Input state:
            - current_query: User's question

        Output state:
            - retrieved_chunks: List of RetrievedChunk
            - error: Error message if retrieval failed
        """
        query = state.get("current_query", "")

        if not query:
            return {
                "retrieved_chunks": [],
                "error": "질문을 입력해주세요."
            }

        # Perform search
        chunks = self.vector_store.search(
            query=query,
            top_k=self.top_k,
            min_score=self.min_score
        )

        if not chunks:
            return {
                "retrieved_chunks": [],
                "error": "관련 정보를 찾을 수 없습니다. 다른 질문을 시도해주세요."
            }

        return {
            "retrieved_chunks": chunks,
            "error": None
        }


class ContextBuilder:
    """Helper class to build context from retrieved chunks."""

    @staticmethod
    def build_context(
        chunks: List[RetrievedChunk],
        max_tokens: int = 8000,
        include_scores: bool = False
    ) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of retrieved chunks
            max_tokens: Approximate max tokens (chars / 4)
            include_scores: Include relevance scores

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4

        for chunk in chunks:
            chunk_text = chunk.to_context_string()
            if include_scores:
                chunk_text = f"[Relevance: {chunk.score:.2f}]\n{chunk_text}"

            if total_chars + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)
