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
from .team import AnalysisTeam
from ..document.sec_downloader import SECDownloader
from ..document.parser import DocumentParser
from ..document.chunker import DocumentChunker
from ..vectorstore.chroma_store import ChromaStore
from ..llm.glm_client import GLMChat, GLMEmbeddings, OpenAIEmbeddings


class AuroraAgent:
    """
    Main AURORA research agent using LangGraph.

    Supports two analysis modes:
    - **standard**: Single-agent RAG pipeline (fast, focused answers)
    - **team**: Multi-agent team analysis with parallel specialist analysts
      (Financial, Risk, Comparative) producing Goldman Sachs-grade reports

    Workflow:
    1. Resolve company name -> ticker/CIK
    2. Download SEC filings
    3. Parse and index documents
    4. Answer questions with citations (standard mode)
       OR run team analysis pipeline (team mode)
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
            llm=self.llm,  # For query optimization
            downloader=self.downloader,  # For company matching
            progress_callback=self.progress_callback  # For progress updates
        )
        self.answerer = AnswererNode(
            llm=self.llm,
            max_context_tokens=16000,
            target_score=8,  # CEO-level quality
            max_iterations=2,  # Refine up to 2 times
            progress_callback=self.progress_callback
        )
        self.citation_validator = CitationValidatorNode(strict=False)

        # Analysis Team (multi-agent)
        self.analysis_team = AnalysisTeam(
            llm=self.llm,
            progress_callback=self.progress_callback,
            max_workers=3,
        )

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

    def _resolve_question_context(self, question: str) -> Dict[str, Any]:
        """
        Shared logic for resolving company and retrieving context from a question.
        Used by both ask() and team_analyze().

        Returns state dict with retrieved_chunks populated.
        """
        # Step 1: Get indexed companies list
        stats = self.vector_store.get_collection_stats()
        indexed_companies = stats.get("indexed_companies", [])

        if not indexed_companies:
            return {
                "current_answer": "",
                "answer_score": 0,
                "error": "인덱스된 문서가 없습니다. 먼저 'aurora research <회사명>'으로 문서를 다운로드해주세요."
            }

        if self.progress_callback:
            self.progress_callback(f"분석 중: 인덱스된 회사 {len(indexed_companies)}개 발견")

        # Step 2: Get indexed tickers mapping
        indexed_tickers = stats.get("indexed_tickers", {})

        if self.progress_callback and indexed_tickers:
            ticker_info = ", ".join([
                f"{name} ({ticker})" for name, ticker in list(indexed_tickers.items())[:3]
            ])
            self.progress_callback(f"인덱스된 티커 정보: {ticker_info}")

        # Step 3: Unified query interpretation
        if self.progress_callback:
            self.progress_callback(
                "질문 교정 및 해석 중: 오타/띄어쓰기 교정 -> 회사명, Form Type, 날짜 정보 추출..."
            )

        interpreted = self.retriever.interpret_query_unified(question)

        if interpreted.get("interpretation_notes") and self.progress_callback:
            self.progress_callback(f"해석 결과: {interpreted['interpretation_notes']}")

        company_or_ticker = interpreted.get("company_name_or_ticker")
        extracted_form_type = interpreted.get("form_type")
        extracted_date = interpreted.get("date_filter")

        # Step 4: Verify company against indexed list
        verified_company_name = None
        matched_company_info = None

        if company_or_ticker:
            from thefuzz import fuzz
            best_match = None
            best_score = 0

            for indexed_company in indexed_companies:
                score = fuzz.partial_ratio(
                    company_or_ticker.lower(),
                    indexed_company.lower()
                )
                if score > best_score:
                    best_score = score
                    best_match = indexed_company

            # Check ticker match
            if indexed_tickers:
                for indexed_company, ticker in indexed_tickers.items():
                    if ticker and company_or_ticker.upper() == ticker.upper():
                        verified_company_name = indexed_company
                        if self.progress_callback:
                            self.progress_callback(
                                f"Ticker match: {ticker} -> {verified_company_name}"
                            )
                        break

            if not verified_company_name and best_score >= 70:
                verified_company_name = best_match
                if self.progress_callback:
                    self.progress_callback(
                        f"Company matched (similarity {best_score}%): {verified_company_name}"
                    )

            # LLM verification for low-confidence matches
            if not verified_company_name and best_score >= 60:
                if self.progress_callback:
                    self.progress_callback("Low similarity - using LLM for precise matching...")
                verified_company_name = self.retriever.verify_company_with_indexed_list(
                    company_or_ticker,
                    indexed_companies,
                    indexed_tickers=indexed_tickers if indexed_tickers else None
                )

        # Fallback fuzzy match
        if not verified_company_name and company_or_ticker:
            from thefuzz import fuzz
            best_match = None
            best_score = 0
            for indexed_company in indexed_companies:
                score = fuzz.partial_ratio(
                    company_or_ticker.lower(),
                    indexed_company.lower()
                )
                if score > best_score:
                    best_score = score
                    best_match = indexed_company
            if best_score >= 60:
                verified_company_name = best_match
            else:
                # Try ticker matching
                if indexed_tickers:
                    for indexed_company, ticker in indexed_tickers.items():
                        if ticker and company_or_ticker.upper() == ticker.upper():
                            verified_company_name = indexed_company
                            break

        # Resolve CompanyInfo from SEC database
        if verified_company_name:
            ticker_to_search = None
            if indexed_tickers and verified_company_name in indexed_tickers:
                ticker_to_search = indexed_tickers[verified_company_name]

            if ticker_to_search:
                candidates = self.downloader.search_company(ticker_to_search, limit=1)
                if candidates:
                    matched_company_info = candidates[0]
            if not matched_company_info:
                candidates = self.downloader.search_company(verified_company_name, limit=1)
                if candidates:
                    matched_company_info = candidates[0]
        else:
            if self.progress_callback:
                self.progress_callback("회사명이 명시되지 않음 - 모든 인덱스된 문서에서 검색")

        # Step 5: Build state and run retriever
        state = create_initial_state()
        state["current_query"] = question
        state["documents_processed"] = True

        if verified_company_name:
            state["indexed_company_name"] = verified_company_name
        if matched_company_info:
            state["company_info"] = matched_company_info

        if extracted_form_type:
            state["filter_form_type"] = extracted_form_type
            if self.progress_callback:
                self.progress_callback(f"Form Type filter: {extracted_form_type}")
        if extracted_date:
            state["filter_date"] = extracted_date
            if self.progress_callback:
                self.progress_callback(f"Date filter: {extracted_date}+")

        # Run retriever
        retriever_result = self.retriever(state)
        state.update(retriever_result)

        # Store resolved metadata for downstream use
        state["_verified_company_name"] = verified_company_name
        state["_matched_company_info"] = matched_company_info

        return state

    def ask(self, question: str, confirm_company: bool = True) -> Dict[str, Any]:
        """
        Ask a question about the indexed documents (standard single-agent mode).

        Args:
            question: User's question
            confirm_company: Whether to confirm company match with user (for CLI)

        Returns:
            State with answer and citations
        """
        state = self._resolve_question_context(question)

        if state.get("error"):
            return state

        # Run answerer + citation validator
        answerer_result = self.answerer(state)
        state.update(answerer_result)
        if state.get("error"):
            return state

        validator_result = self.citation_validator(state)
        state.update(validator_result)
        return state

    def team_analyze(self, question: str) -> Dict[str, Any]:
        """
        Run multi-agent team analysis (Goldman Sachs-grade).

        Deploys 3 specialist analysts in parallel:
        - Financial Analyst: Metrics extraction, ratio analysis, financial health scoring
        - Risk Analyst: Risk identification, categorization, quantitative scoring
        - Comparative Analyst: Cross-period trends, trajectory, management guidance

        Results are synthesized by a Report Synthesizer (MD-level) into a
        unified executive research report.

        Args:
            question: User's question

        Returns:
            Dict with team report, risk rating, key metrics, analysts used, etc.
        """
        if self.progress_callback:
            self.progress_callback("=" * 50)
            self.progress_callback("AURORA Team Analysis Mode")
            self.progress_callback("Deploying: Financial, Risk, Comparative Analysts")
            self.progress_callback("=" * 50)

        state = self._resolve_question_context(question)

        if state.get("error"):
            return state

        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {
                "team_report": "",
                "error": "관련 문서를 찾을 수 없습니다.",
            }

        # Build metadata for analysts
        company_name = "Unknown"
        verified = state.get("_verified_company_name")
        matched = state.get("_matched_company_info")
        if matched:
            company_name = matched.name
        elif verified:
            company_name = verified

        metadata = {
            "company_name": company_name,
            "company_info": matched,
            "total_chunks": len(chunks),
        }

        # Run parallel team analysis
        team_result = self.analysis_team.run_analysis(
            question=question,
            chunks=chunks,
            metadata=metadata,
            max_context_tokens=16000,
        )

        # Map to state
        state["analysis_mode"] = "team"
        state["team_report"] = team_result.get("report", "")
        state["current_answer"] = team_result.get("report", "")
        state["risk_rating"] = team_result.get("risk_rating", "N/A")
        state["risk_flags"] = team_result.get("risk_flags", [])
        state["key_metrics"] = team_result.get("key_metrics", {})
        state["analysis_confidence"] = team_result.get("confidence", 0.0)
        state["analysts_used"] = team_result.get("analysts_used", [])
        state["error"] = team_result.get("error")

        return state

    def get_stats(self) -> Dict[str, Any]:
        """Get current vector store statistics."""
        return self.vector_store.get_collection_stats()
