"""Retriever node - searches vector store for relevant chunks."""
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...vectorstore.chroma_store import ChromaStore, RetrievedChunk


class RetrieverNode:
    """
    Node that retrieves relevant document chunks for a query.

    Features:
    - Query optimization using LLM
    - Multi-query search for better recall
    - Semantic search using embeddings
    - Metadata filtering (form type, section)
    - Relevance scoring
    """

    QUERY_OPTIMIZER_PROMPT = """You are a search query optimizer for SEC financial documents.

CRITICAL RULES:
1. NEVER include company names (e.g., "Vertical", "Apple", "Tesla") in search queries
2. Focus ONLY on financial concepts and SEC terminology
3. Generate 2-3 short, focused English search queries

Financial terms to use:
- going concern, substantial doubt, ability to continue
- liquidity risk, cash burn, funding requirements
- net losses, operating cash flows, capital requirements
- risk factors, material uncertainty

Output ONLY the search queries, one per line. No company names. No explanations."""

    def __init__(
        self,
        vector_store: ChromaStore,
        top_k: int = 10,
        min_score: float = 0.3,
        llm: Optional[BaseChatModel] = None
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.llm = llm

    def _optimize_query(self, query: str) -> List[str]:
        """Use LLM to generate optimized search queries."""
        if not self.llm:
            return [query]
        
        try:
            messages = [
                SystemMessage(content=self.QUERY_OPTIMIZER_PROMPT),
                HumanMessage(content=f"User question: {query}")
            ]
            response = self.llm.invoke(messages)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            # Always include original query as fallback
            if query not in queries:
                queries.append(query)
            return queries[:3]  # Max 3 queries
        except Exception:
            return [query]

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

        # Optimize query using LLM
        search_queries = self._optimize_query(query)
        
        # Perform multi-query search
        all_chunks = {}
        for sq in search_queries:
            chunks = self.vector_store.search(
                query=sq,
                top_k=self.top_k,
                min_score=self.min_score
            )
            for chunk in chunks:
                # Use content hash as key to deduplicate
                key = hash(chunk.content)
                if key not in all_chunks or chunk.score > all_chunks[key].score:
                    all_chunks[key] = chunk
        
        # Sort by score and take top results
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
        result_chunks = sorted_chunks[:self.top_k]

        if not result_chunks:
            return {
                "retrieved_chunks": [],
                "error": "관련 정보를 찾을 수 없습니다. 다른 질문을 시도해주세요."
            }

        return {
            "retrieved_chunks": result_chunks,
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
