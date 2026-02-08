"""Base analyst class for the multi-agent team."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structured output from an analyst agent."""

    analyst_type: str
    sections: Dict[str, str]  # section_name -> content
    confidence: float  # 0.0 - 1.0
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAnalyst(ABC):
    """
    Base class for all specialist analyst agents.

    Each analyst receives the same context (retrieved SEC document chunks)
    and produces a structured AnalysisResult focused on their domain.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.progress_callback = progress_callback

    def _report(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)

    @property
    @abstractmethod
    def analyst_type(self) -> str:
        """Return the analyst type identifier."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this analyst."""

    @abstractmethod
    def analyze(self, question: str, context: str, metadata: Dict[str, Any]) -> AnalysisResult:
        """
        Run the analysis.

        Args:
            question: User's original question
            context: Formatted SEC document chunks
            metadata: Additional context (company info, form types, etc.)

        Returns:
            AnalysisResult with structured analysis
        """

    def _invoke_llm(self, system: str, user: str) -> str:
        """Helper to invoke LLM with system + user messages."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"[{self.analyst_type}] LLM invocation failed: {e}")
            return ""
