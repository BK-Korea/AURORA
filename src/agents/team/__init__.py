"""Multi-Agent Analysis Team for Goldman Sachs-level SEC research."""
from .base_analyst import BaseAnalyst, AnalysisResult
from .financial_analyst import FinancialAnalyst
from .risk_analyst import RiskAnalyst
from .comparative_analyst import ComparativeAnalyst
from .report_synthesizer import ReportSynthesizer
from .team_orchestrator import AnalysisTeam

__all__ = [
    "BaseAnalyst",
    "AnalysisResult",
    "FinancialAnalyst",
    "RiskAnalyst",
    "ComparativeAnalyst",
    "ReportSynthesizer",
    "AnalysisTeam",
]
