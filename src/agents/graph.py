"""LangGraph workflow for AURORA research agent."""
from typing import Dict, Any, Optional, Callable, Literal
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from .state import create_initial_state
from .nodes import (
    CompanyResolverNode,
    DocumentFetcherNode,
    RetrieverNode,
    AnswererNode,
    CitationValidatorNode,
)
from ..document.sec_downloader import SECDownloader
from ..document.parser import DocumentParser
from ..document.chunker import DocumentChunker
from ..vectorstore.chroma_store import ChromaStore
from ..llm.glm_client import GLMChat, GLMEmbeddings, OpenAIEmbeddings


class AuroraAgent:
    """
    Main AURORA research agent using LangGraph.

    Workflow:
    1. Resolve company name → ticker/CIK
    2. Download SEC filings
    3. Parse and index documents
    4. Answer questions with citations
    5. Validate citations
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        data_dir: Path = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        openai_api_key: Optional[str] = None,
        embedding_provider: str = "openai",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.data_dir = data_dir or Path("data")
        self.progress_callback = progress_callback
        self.openai_api_key = openai_api_key
        self.embedding_provider = embedding_provider

        # Initialize components
        self._init_components()

        # Build graph
        self.graph = self._build_graph()

    def _init_components(self):
        """Initialize all agent components."""
        # LLM
        self.llm = GLMChat(
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
            model="glm-4.7",
            temperature=0.1,  # Low for factual accuracy
            timeout=180,  # Increased for detailed answers
        )

        # Embeddings - use OpenAI by default (more reliable)
        if self.embedding_provider == "openai" and self.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                api_key=SecretStr(self.openai_api_key),
                model="text-embedding-3-small",
            )
        else:
            self.embeddings = GLMEmbeddings(
                api_key=SecretStr(self.api_key),
                base_url=self.base_url,
                model="embedding-2",
            )

        # Document processing
        self.downloader = SECDownloader(data_dir=self.data_dir)
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Vector store
        self.vector_store = ChromaStore(
            persist_dir=self.data_dir / "vectordb",
            embeddings=self.embeddings,
            collection_name="sec_documents"
        )

        # Nodes
        self.company_resolver = CompanyResolverNode(
            downloader=self.downloader,
            llm=self.llm
        )
        self.document_fetcher = DocumentFetcherNode(
            downloader=self.downloader,
            parser=self.parser,
            chunker=self.chunker,
            vector_store=self.vector_store,
            progress_callback=self.progress_callback
        )
        self.retriever = RetrieverNode(
            vector_store=self.vector_store,
            top_k=20,  # Increased for comprehensive answers
            min_score=0.2,  # Lower threshold for more results
            llm=self.llm  # For query optimization
        )
        self.answerer = AnswererNode(
            llm=self.llm,
            max_context_tokens=16000,
            target_score=8,  # CEO-level quality
            max_iterations=2,  # Refine up to 2 times
            progress_callback=self.progress_callback
        )
        self.citation_validator = CitationValidatorNode(strict=False)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Define state schema
        from typing import TypedDict, List, Optional
        from ..document.sec_downloader import CompanyInfo, SECFiling
        from ..vectorstore.chroma_store import RetrievedChunk

        class State(TypedDict):
            company_query: str
            company_info: Optional[CompanyInfo]
            company_confirmed: bool
            years_requested: int
            filings: List[SECFiling]
            documents_processed: bool
            chunks_indexed: int
            current_query: str
            retrieved_chunks: List[RetrievedChunk]
            current_answer: str
            citations_valid: bool
            error: Optional[str]
            session_active: bool

        # Create graph
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("resolve_company", self.company_resolver)
        workflow.add_node("fetch_documents", self.document_fetcher)
        workflow.add_node("retrieve", self.retriever)
        workflow.add_node("answer", self.answerer)
        workflow.add_node("validate_citations", self.citation_validator)

        # Define edges
        workflow.set_entry_point("resolve_company")

        # Conditional: company resolved?
        def should_fetch(state: State) -> Literal["fetch_documents", "end"]:
            if state.get("company_info") and not state.get("documents_processed"):
                return "fetch_documents"
            return "end"

        workflow.add_conditional_edges(
            "resolve_company",
            should_fetch,
            {
                "fetch_documents": "fetch_documents",
                "end": END
            }
        )

        workflow.add_edge("fetch_documents", END)
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", "validate_citations")
        workflow.add_edge("validate_citations", END)

        return workflow.compile()

    def resolve_company(self, company_query: str) -> Dict[str, Any]:
        """
        Resolve company name and prepare for document fetching.

        Args:
            company_query: Company name or ticker

        Returns:
            State with resolved company info
        """
        initial_state = create_initial_state()
        initial_state["company_query"] = company_query

        result = self.graph.invoke(initial_state)
        return result

    def fetch_documents(
        self,
        company_query: str,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        Resolve company and fetch documents.

        Args:
            company_query: Company name or ticker
            years: Number of years of filings

        Returns:
            State with fetched documents
        """
        initial_state = create_initial_state()
        initial_state["company_query"] = company_query
        initial_state["years_requested"] = years

        result = self.graph.invoke(initial_state)
        return result

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the indexed documents.

        Args:
            question: User's question

        Returns:
            State with answer and citations
        """
        # Create minimal state for Q&A
        state = create_initial_state()
        state["current_query"] = question
        state["documents_processed"] = True

        # Run through Q&A nodes manually (update state, don't replace)
        retriever_result = self.retriever(state)
        state.update(retriever_result)
        if state.get("error"):
            return state

        answerer_result = self.answerer(state)
        state.update(answerer_result)
        if state.get("error"):
            return state

        validator_result = self.citation_validator(state)
        state.update(validator_result)
        return state

    def get_stats(self) -> Dict[str, Any]:
        """Get current vector store statistics."""
        return self.vector_store.get_collection_stats()
