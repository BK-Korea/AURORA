"""Team Orchestrator - Runs specialist analysts in parallel and synthesizes results."""
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable
from langchain_core.language_models import BaseChatModel

from .base_analyst import BaseAnalyst, AnalysisResult
from .financial_analyst import FinancialAnalyst
from .risk_analyst import RiskAnalyst
from .comparative_analyst import ComparativeAnalyst
from .strategy_analyst import StrategyAnalyst
from .report_synthesizer import ReportSynthesizer
from ...vectorstore.chroma_store import RetrievedChunk

logger = logging.getLogger(__name__)


class AnalysisTeam:
    """
    Orchestrates a team of specialist analyst agents for comprehensive SEC analysis.

    Architecture (GS + McKinsey Dual-Perspective):
    ┌──────────────────────────────────────────────────────────────┐
    │                     Analysis Team                            │
    │                                                              │
    │  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
    │  │  Strategy    │ │ Financial│ │   Risk   │ │ Comparative│  │
    │  │  (McKinsey)  │ │   (GS)   │ │  (GS)   │ │   (GS)     │  │
    │  └──────┬──────┘ └────┬─────┘ └────┬─────┘ └──────┬─────┘  │
    │         │             │            │               │        │
    │         └─────────┬───┘────────────┘───────────────┘        │
    │                   │                                         │
    │          ┌────────▼────────────┐                            │
    │          │  Report Synthesizer │                            │
    │          │  (Strategy+Finance) │                            │
    │          └─────────────────────┘                            │
    └──────────────────────────────────────────────────────────────┘

    4 analysts run in parallel, then the synthesizer produces an
    integrated strategy + finance executive report.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        progress_callback: Optional[Callable[[str], None]] = None,
        max_workers: int = 4,
    ):
        self.llm = llm
        self.progress_callback = progress_callback
        self.max_workers = max_workers

        # Initialize specialist analysts (4 agents)
        self.analysts: List[BaseAnalyst] = [
            StrategyAnalyst(llm=llm, progress_callback=progress_callback),
            FinancialAnalyst(llm=llm, progress_callback=progress_callback),
            RiskAnalyst(llm=llm, progress_callback=progress_callback),
            ComparativeAnalyst(llm=llm, progress_callback=progress_callback),
        ]

        # Report synthesizer (GS + McKinsey integrated)
        self.synthesizer = ReportSynthesizer(
            llm=llm, progress_callback=progress_callback
        )

    def _report(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)

    def run_analysis(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        metadata: Dict[str, Any],
        max_context_tokens: int = 16000,
    ) -> Dict[str, Any]:
        """
        Run the full analysis team pipeline.

        Args:
            question: User's question
            chunks: Retrieved document chunks
            metadata: Company info and other context
            max_context_tokens: Max tokens for context window

        Returns:
            Dict with synthesized report and all analyst results
        """
        # Build context from chunks
        context = self._build_context(chunks, max_context_tokens)

        if not context:
            return {
                "report": "",
                "error": "No document context available for analysis.",
                "analyst_results": [],
            }

        self._report(
            f"Analysis Team: Deploying {len(self.analysts)} specialist analysts in parallel..."
        )

        # Run all analysts in parallel using ThreadPoolExecutor
        analyst_results = self._run_analysts_parallel(question, context, metadata)

        successful = [r for r in analyst_results if not r.error]
        failed = [r for r in analyst_results if r.error]

        self._report(
            f"Analysis Team: {len(successful)}/{len(self.analysts)} analysts completed successfully."
        )

        if failed:
            for r in failed:
                self._report(f"  Warning: {r.analyst_type} failed - {r.error}")

        if not successful:
            return {
                "report": "",
                "error": "All analyst agents failed. Please try again.",
                "analyst_results": analyst_results,
            }

        # Synthesize all results into executive report
        synthesis = self.synthesizer.synthesize(
            question=question,
            analyst_results=analyst_results,
            metadata=metadata,
        )

        return {
            "report": synthesis["report"],
            "risk_rating": synthesis.get("risk_rating", "N/A"),
            "risk_flags": synthesis.get("risk_flags", []),
            "confidence": synthesis.get("confidence", 0.0),
            "key_metrics": synthesis.get("key_metrics", {}),
            "analysts_used": synthesis.get("analysts_used", []),
            "analyst_results": analyst_results,
            "error": None,
        }

    def _run_analysts_parallel(
        self, question: str, context: str, metadata: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Run all analysts in parallel using ThreadPoolExecutor."""
        results: List[AnalysisResult] = []

        def run_single(analyst: BaseAnalyst) -> AnalysisResult:
            try:
                return analyst.analyze(question, context, metadata)
            except Exception as e:
                logger.error(f"Analyst {analyst.analyst_type} failed: {e}")
                return AnalysisResult(
                    analyst_type=analyst.analyst_type,
                    sections={},
                    confidence=0.0,
                    error=str(e),
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_analyst = {
                executor.submit(run_single, analyst): analyst
                for analyst in self.analysts
            }

            for future in concurrent.futures.as_completed(future_to_analyst):
                analyst = future_to_analyst[future]
                try:
                    result = future.result(timeout=300)  # 5 min timeout per analyst
                    results.append(result)
                    self._report(
                        f"  [{result.analyst_type}] Complete (confidence: {result.confidence:.0%})"
                    )
                except concurrent.futures.TimeoutError:
                    results.append(
                        AnalysisResult(
                            analyst_type=analyst.analyst_type,
                            sections={},
                            confidence=0.0,
                            error="Analysis timed out (5 min limit)",
                        )
                    )
                except Exception as e:
                    results.append(
                        AnalysisResult(
                            analyst_type=analyst.analyst_type,
                            sections={},
                            confidence=0.0,
                            error=str(e),
                        )
                    )

        return results

    @staticmethod
    def _build_context(
        chunks: List[RetrievedChunk], max_tokens: int = 16000
    ) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough token-to-char ratio

        for chunk in chunks:
            chunk_text = chunk.to_context_string()
            if total_chars + len(chunk_text) > max_chars:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)
