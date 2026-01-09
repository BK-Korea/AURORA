"""Agent state definitions for LangGraph."""
from typing import List, Optional, Dict, Any, Annotated
from dataclasses import dataclass, field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from ..document.sec_downloader import CompanyInfo, SECFiling
from ..vectorstore.chroma_store import RetrievedChunk


@dataclass
class AgentState:
    """
    State for the AURORA research agent.

    Tracks the entire research session including:
    - Company identification
    - Downloaded documents
    - Current query and retrieved context
    - Generated answers with citations
    """
    # Company information
    company_query: str = ""
    company_info: Optional[CompanyInfo] = None
    company_confirmed: bool = False

    # Document status
    years_requested: int = 3
    filings: List[SECFiling] = field(default_factory=list)
    documents_processed: bool = False
    chunks_indexed: int = 0

    # Query handling
    current_query: str = ""
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)

    # Answer generation
    current_answer: str = ""
    citations_valid: bool = False

    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)

    # Error handling
    error: Optional[str] = None

    # Session metadata
    session_active: bool = True


def create_initial_state() -> Dict[str, Any]:
    """Create initial state dictionary for LangGraph."""
    return {
        "company_query": "",
        "company_info": None,
        "company_confirmed": False,
        "years_requested": 3,
        "filings": [],
        "documents_processed": False,
        "chunks_indexed": 0,
        "current_query": "",
        "retrieved_chunks": [],
        "current_answer": "",
        "citations_valid": False,
        "messages": [],
        "error": None,
        "session_active": True,
    }
