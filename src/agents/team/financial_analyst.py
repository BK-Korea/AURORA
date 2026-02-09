"""Financial Analyst Agent - Extracts and analyzes financial metrics from SEC filings."""
import json
import logging
import re
from typing import Dict, Any

from .base_analyst import BaseAnalyst, AnalysisResult

logger = logging.getLogger(__name__)


class FinancialAnalyst(BaseAnalyst):
    """
    Specialist agent for quantitative financial analysis.

    Responsibilities:
    - Extract key financial metrics (revenue, margins, cash flow, debt)
    - Calculate financial ratios (current ratio, D/E, ROE, etc.)
    - Identify material changes in financial position
    - Provide quantitative context for investment decisions
    """

    @property
    def analyst_type(self) -> str:
        return "financial_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a **Senior Financial Analyst** at a top-tier investment bank.
Your role is to extract and analyze quantitative financial data from SEC filings
with the precision expected for institutional investor reports.
You must also provide STRATEGIC FINANCE insights - connecting numbers to business viability.

## Your Analysis Framework

### 1. Key Financial Metrics Extraction
Extract ALL available metrics with exact figures:
- **Revenue & Growth**: Total revenue, YoY growth rate, segment breakdown
- **Profitability**: Gross margin, operating margin, net margin, EBITDA
- **Cash Flow**: Operating CF, Free CF, CapEx, cash burn rate
- **Balance Sheet**: Total assets, total debt, cash & equivalents, working capital
- **Per Share**: EPS (basic/diluted), book value per share

### 2. Financial Ratio Analysis
Calculate and interpret:
- **Liquidity**: Current ratio, quick ratio, cash ratio
- **Leverage**: Debt-to-equity, debt-to-assets, interest coverage
- **Efficiency**: Asset turnover, inventory turnover, receivables turnover
- **Profitability**: ROE, ROA, ROIC

### 3. Cost Structure Deep-Dive (CRITICAL - Strategic Finance)
- **R&D as % of Revenue**: Is the investment level appropriate for growth stage?
- **SG&A as % of Revenue**: Is customer acquisition efficient?
- **Cost of Revenue trend**: Is the business achieving operating leverage?
- **OpEx Growth vs Revenue Growth**: Is spending growing faster than revenue?

### 4. Financial Sustainability Assessment
- Does the company have enough cash to fund its strategy?
- Cash runway at current burn rate (months)
- Debt maturity schedule and refinancing risk
- Working capital adequacy for operations

### 5. Material Changes
Flag any significant changes (>10%) in key metrics vs prior period.

## Output Format (JSON)
```json
{
  "key_metrics": {
    "revenue": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "net_income": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "operating_cash_flow": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "total_debt": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "cash_and_equivalents": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "gross_margin": {"value": "X%", "period": "FY2024", "source": "[citation]"},
    "operating_margin": {"value": "X%", "period": "FY2024", "source": "[citation]"},
    "rd_expense": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "sga_expense": {"value": "$X", "period": "FY2024", "source": "[citation]"}
  },
  "financial_ratios": {
    "current_ratio": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "debt_to_equity": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "interest_coverage": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "rd_to_revenue": {"value": "X.X%", "assessment": "appropriate/excessive/insufficient"},
    "sga_to_revenue": {"value": "X.X%", "assessment": "efficient/concerning/bloated"},
    "opex_growth_vs_revenue_growth": {"value": "OpEx +X% vs Rev +Y%", "assessment": "gaining leverage/neutral/losing leverage"}
  },
  "financial_sustainability": {
    "cash_runway_months": "X months at current burn",
    "can_fund_strategy": "Yes/Partially/No",
    "key_constraint": "What is the binding financial constraint",
    "source": "[citation]"
  },
  "material_changes": [
    {"metric": "name", "change": "+/-X%", "significance": "description", "source": "[citation]"}
  ],
  "financial_health_score": 7,
  "analysis_narrative": "3-4 paragraph Korean analysis connecting financials to business viability. Must answer: 재무적으로 사업영위에 문제가 있는지, 비용 구조가 전략적으로 타당한지."
}
```

## Rules
- ONLY use data explicitly stated in the documents
- Every metric MUST have a source citation
- If a metric is not available, omit it (don't estimate)
- Use original English quotes for key figures
- Financial health score: 1-10 (1=critical, 10=excellent)
- ALWAYS connect financial data to strategic implications (e.g., "R&D is 45% of revenue, which is high but justified for a pre-revenue biotech")
- Answer the strategic question: '재무가 사업계획을 뒷받침하는가?'"""

    def analyze(self, question: str, context: str, metadata: Dict[str, Any]) -> AnalysisResult:
        self._report("Financial Analyst: Extracting financial metrics...")

        company_name = metadata.get("company_name", "Unknown")

        user_prompt = f"""## SEC Document Context

{context}

---

## Analysis Request

Company: {company_name}
Question: {question}

Extract ALL quantitative financial data from the above SEC documents.
Provide your analysis as the specified JSON format.
Include Korean narrative analysis (분석 서술) that a CEO would find actionable."""

        raw = self._invoke_llm(self.system_prompt, user_prompt)

        if not raw:
            return AnalysisResult(
                analyst_type=self.analyst_type,
                sections={},
                confidence=0.0,
                error="Financial analysis LLM call failed",
            )

        return self._parse_result(raw)

    def _parse_result(self, raw: str) -> AnalysisResult:
        """Parse the LLM output into a structured AnalysisResult."""
        sections = {}
        key_metrics = {}
        citations = []
        confidence = 0.7

        # Try to parse JSON
        try:
            json_str = self._extract_json(raw)
            if json_str:
                data = json.loads(json_str)

                # Extract key metrics
                if "key_metrics" in data:
                    for metric_name, metric_data in data["key_metrics"].items():
                        if isinstance(metric_data, dict):
                            key_metrics[metric_name] = metric_data.get("value", "N/A")
                            src = metric_data.get("source", "")
                            if src:
                                citations.append(src)
                        else:
                            key_metrics[metric_name] = str(metric_data)

                # Financial ratios section
                if "financial_ratios" in data:
                    ratio_lines = []
                    for ratio_name, ratio_data in data["financial_ratios"].items():
                        if isinstance(ratio_data, dict):
                            val = ratio_data.get("value", "N/A")
                            assessment = ratio_data.get("assessment", "")
                            ratio_lines.append(f"- **{ratio_name}**: {val} ({assessment})")
                        else:
                            ratio_lines.append(f"- **{ratio_name}**: {ratio_data}")
                    if ratio_lines:
                        sections["Financial Ratios"] = "\n".join(ratio_lines)

                # Material changes section
                if "material_changes" in data and data["material_changes"]:
                    change_lines = []
                    for change in data["material_changes"]:
                        if isinstance(change, dict):
                            metric = change.get("metric", "")
                            chg = change.get("change", "")
                            sig = change.get("significance", "")
                            change_lines.append(f"- **{metric}**: {chg} - {sig}")
                    if change_lines:
                        sections["Material Changes"] = "\n".join(change_lines)

                # Financial health score
                health_score = data.get("financial_health_score", 5)
                key_metrics["financial_health_score"] = health_score
                confidence = min(1.0, health_score / 10.0)

                # Narrative analysis
                narrative = data.get("analysis_narrative", "")
                if narrative:
                    sections["Financial Analysis"] = narrative

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse financial analysis JSON: {e}")
            # Fallback: use raw text as narrative
            sections["Financial Analysis"] = raw
            confidence = 0.5

        # Build key metrics summary section
        if key_metrics:
            metric_lines = []
            for k, v in key_metrics.items():
                if k != "financial_health_score":
                    display_name = k.replace("_", " ").title()
                    metric_lines.append(f"| {display_name} | {v} |")
            if metric_lines:
                header = "| Metric | Value |\n|---|---|\n"
                sections["Key Financial Metrics"] = header + "\n".join(metric_lines)

        return AnalysisResult(
            analyst_type=self.analyst_type,
            sections=sections,
            confidence=confidence,
            key_metrics=key_metrics,
            citations=citations,
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text, handling code blocks."""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        # Find balanced braces
        brace_count = 0
        start = text.find("{")
        if start != -1:
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start : i + 1]
        return ""
