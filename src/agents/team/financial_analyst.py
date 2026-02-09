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
You must apply advanced financial analysis frameworks and provide STRATEGIC FINANCE
insights - connecting numbers to business viability and financial distress signals.

## Your Analysis Framework

### 1. Key Financial Metrics Extraction
Extract ALL available metrics with exact figures:
- **Revenue & Growth**: Total revenue, YoY growth rate, segment breakdown
- **Profitability**: Gross margin, operating margin, net margin, EBITDA
- **Cash Flow**: Operating CF, Free CF, CapEx, cash burn rate
- **Balance Sheet**: Total assets, total liabilities, total equity, total debt,
  cash & equivalents, working capital, current assets, current liabilities
- **Per Share**: EPS (basic/diluted), book value per share

### 2. DuPont Analysis (CRITICAL - ROE Decomposition)
Decompose Return on Equity into its three components:
- **Net Profit Margin** = Net Income / Revenue (profitability efficiency)
- **Asset Turnover** = Revenue / Total Assets (asset utilization efficiency)
- **Equity Multiplier** = Total Assets / Shareholders' Equity (financial leverage)
- **ROE** = Net Profit Margin x Asset Turnover x Equity Multiplier
Identify which component drives ROE the most and whether it's sustainable.
Is high ROE driven by genuine profitability or excessive leverage?

### 3. Altman Z-Score (Bankruptcy Prediction)
Calculate the Z-Score where data permits:
- **Z = 1.2*(Working Capital/Total Assets) + 1.4*(Retained Earnings/Total Assets)
  + 3.3*(EBIT/Total Assets) + 0.6*(Market Cap or Equity/Total Liabilities)
  + 1.0*(Revenue/Total Assets)**
- Z > 2.99: Safe Zone (low bankruptcy risk)
- 1.81 < Z < 2.99: Grey Zone (moderate risk)
- Z < 1.81: Distress Zone (high bankruptcy risk)
If exact inputs unavailable, use available proxies and note limitations.

### 4. Operating Leverage Analysis
- **Contribution Margin**: (Revenue - Variable Costs) / Revenue
- **Degree of Operating Leverage (DOL)**: % Change in EBIT / % Change in Revenue
- Is the business model high or low operating leverage?
- What are the implications for profitability scaling?
- **Breakeven Analysis**: Estimated fixed costs vs contribution margin
- At what revenue level does the company break even?

### 5. Financial Ratio Analysis
Calculate and interpret:
- **Liquidity**: Current ratio, quick ratio, cash ratio
- **Leverage**: Debt-to-equity, debt-to-assets, interest coverage, net debt/EBITDA
- **Efficiency**: Asset turnover, inventory turnover, receivables turnover, cash conversion cycle
- **Profitability**: ROE (DuPont decomposed), ROA, ROIC

### 6. Cost Structure Deep-Dive (Strategic Finance)
- **R&D as % of Revenue**: Investment level vs growth stage appropriateness
- **SG&A as % of Revenue**: Customer acquisition efficiency
- **Cost of Revenue trend**: Operating leverage achievement
- **OpEx Growth vs Revenue Growth**: Spending growth relative to revenue growth
- **Fixed vs Variable Cost Mix**: Business model scalability assessment

### 7. Financial Sustainability Assessment
- Cash runway at current burn rate (months)
- Debt maturity schedule and refinancing risk
- Working capital adequacy for operations
- Free cash flow generation capability
- Dividend sustainability (if applicable)

### 8. Earnings Quality Assessment
- **Accrual Ratio**: (Net Income - Operating Cash Flow) / Total Assets
  High positive accrual ratio = lower earnings quality (earnings not backed by cash)
- **Cash Flow to Net Income ratio**: OCF / Net Income (>1.0 is healthy)
- Revenue recognition policies: any aggressive practices?
- One-time items vs recurring income

### 9. Material Changes
Flag any significant changes (>10%) in key metrics vs prior period.

## Output Format (JSON)
```json
{
  "key_metrics": {
    "revenue": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "net_income": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "ebit": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "ebitda": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "operating_cash_flow": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "free_cash_flow": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "total_assets": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "total_equity": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "total_debt": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "cash_and_equivalents": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "working_capital": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "gross_margin": {"value": "X%", "period": "FY2024", "source": "[citation]"},
    "operating_margin": {"value": "X%", "period": "FY2024", "source": "[citation]"},
    "rd_expense": {"value": "$X", "period": "FY2024", "source": "[citation]"},
    "sga_expense": {"value": "$X", "period": "FY2024", "source": "[citation]"}
  },
  "dupont_analysis": {
    "net_profit_margin": {"value": "X%", "source": "[citation]"},
    "asset_turnover": {"value": "X.Xx", "source": "[citation]"},
    "equity_multiplier": {"value": "X.Xx", "source": "[citation]"},
    "roe": {"value": "X%", "source": "[citation]"},
    "primary_driver": "Margin/Turnover/Leverage",
    "sustainability_assessment": "Is the ROE driver sustainable? Why?"
  },
  "altman_z_score": {
    "score": "X.XX",
    "zone": "Safe/Grey/Distress",
    "components": {
      "working_capital_to_assets": "X.XX",
      "retained_earnings_to_assets": "X.XX",
      "ebit_to_assets": "X.XX",
      "equity_to_liabilities": "X.XX",
      "revenue_to_assets": "X.XX"
    },
    "interpretation": "What the Z-Score implies about bankruptcy risk",
    "limitations": "Note any data gaps or proxy usage"
  },
  "operating_leverage": {
    "dol_estimate": "X.Xx",
    "fixed_variable_mix": "High fixed / Low fixed",
    "breakeven_assessment": "Above/Below/Near breakeven",
    "scalability": "High/Moderate/Low",
    "interpretation": "What operating leverage means for this company"
  },
  "financial_ratios": {
    "current_ratio": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "debt_to_equity": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "interest_coverage": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "net_debt_to_ebitda": {"value": "X.X", "assessment": "healthy/concerning/critical"},
    "cash_conversion_cycle": {"value": "X days", "assessment": "efficient/average/slow"},
    "rd_to_revenue": {"value": "X.X%", "assessment": "appropriate/excessive/insufficient"},
    "sga_to_revenue": {"value": "X.X%", "assessment": "efficient/concerning/bloated"},
    "opex_growth_vs_revenue_growth": {"value": "OpEx +X% vs Rev +Y%", "assessment": "gaining leverage/neutral/losing leverage"}
  },
  "earnings_quality": {
    "accrual_ratio": {"value": "X.XX", "assessment": "high quality/moderate/low quality"},
    "ocf_to_net_income": {"value": "X.Xx", "assessment": "cash-backed/moderate/accrual-heavy"},
    "revenue_recognition_flag": "Conservative/Moderate/Aggressive",
    "one_time_items": "Description of significant one-time items if any"
  },
  "financial_sustainability": {
    "cash_runway_months": "X months at current burn",
    "can_fund_strategy": "Yes/Partially/No",
    "key_constraint": "What is the binding financial constraint",
    "fcf_generation": "Positive and growing / Positive but declining / Negative",
    "source": "[citation]"
  },
  "material_changes": [
    {"metric": "name", "change": "+/-X%", "significance": "description", "source": "[citation]"}
  ],
  "financial_health_score": 7,
  "analysis_narrative": "4-5 paragraph Korean analysis. Must cover: DuPont ROE 분해 결과, Altman Z-Score 해석, 영업레버리지 시사점, 이익의 질(Earnings Quality), 재무적으로 사업영위에 문제가 있는지, 비용 구조가 전략적으로 타당한지. 최종 판단: 재무가 사업계획을 뒷받침하는가?"
}
```

## Rules
- ONLY use data explicitly stated in the documents
- Every metric MUST have a source citation
- If a metric is not available, omit it (don't estimate wildly)
- For DuPont and Z-Score: if some inputs are missing, calculate with available data and note limitations
- Use original English quotes for key figures
- Financial health score: 1-10 (1=critical, 10=excellent)
- ALWAYS connect financial data to strategic implications
- Earnings quality is as important as earnings level - flag accrual/cash divergence
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
        risk_flags = []
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

                # DuPont Analysis section
                dupont = data.get("dupont_analysis", {})
                if dupont:
                    dp_lines = [
                        "| Component | Value |",
                        "|---|---|",
                        f"| Net Profit Margin | {dupont.get('net_profit_margin', {}).get('value', 'N/A')} |",
                        f"| Asset Turnover | {dupont.get('asset_turnover', {}).get('value', 'N/A')} |",
                        f"| Equity Multiplier | {dupont.get('equity_multiplier', {}).get('value', 'N/A')} |",
                        f"| **ROE (DuPont)** | **{dupont.get('roe', {}).get('value', 'N/A')}** |",
                    ]
                    driver = dupont.get("primary_driver", "")
                    sustain = dupont.get("sustainability_assessment", "")
                    if driver:
                        dp_lines.append(f"\n**Primary ROE Driver**: {driver}")
                    if sustain:
                        dp_lines.append(f"**Sustainability**: {sustain}")
                    sections["DuPont ROE Analysis"] = "\n".join(dp_lines)
                    key_metrics["roe_dupont"] = dupont.get("roe", {}).get("value", "N/A")
                    key_metrics["roe_primary_driver"] = driver
                    # Collect citations from dupont components
                    for comp in ["net_profit_margin", "asset_turnover", "equity_multiplier", "roe"]:
                        src = dupont.get(comp, {}).get("source", "")
                        if src:
                            citations.append(src)

                # Altman Z-Score section
                zscore = data.get("altman_z_score", {})
                if zscore:
                    score_val = zscore.get("score", "N/A")
                    zone = zscore.get("zone", "N/A")
                    interp = zscore.get("interpretation", "")
                    limits = zscore.get("limitations", "")
                    components = zscore.get("components", {})

                    z_lines = [f"**Altman Z-Score**: {score_val} ({zone} Zone)"]
                    if components:
                        z_lines.append("\n| Component | Value |")
                        z_lines.append("|---|---|")
                        comp_names = {
                            "working_capital_to_assets": "Working Capital / Assets",
                            "retained_earnings_to_assets": "Retained Earnings / Assets",
                            "ebit_to_assets": "EBIT / Assets",
                            "equity_to_liabilities": "Equity / Liabilities",
                            "revenue_to_assets": "Revenue / Assets",
                        }
                        for comp_key, comp_label in comp_names.items():
                            val = components.get(comp_key, "N/A")
                            z_lines.append(f"| {comp_label} | {val} |")
                    if interp:
                        z_lines.append(f"\n**Interpretation**: {interp}")
                    if limits:
                        z_lines.append(f"**Limitations**: {limits}")
                    sections["Altman Z-Score (Bankruptcy Risk)"] = "\n".join(z_lines)
                    key_metrics["altman_z_score"] = score_val
                    key_metrics["altman_zone"] = zone
                    if zone == "Distress":
                        risk_flags.append(f"[FINANCIAL] Altman Z-Score in Distress Zone: {score_val}")

                # Operating Leverage section
                op_lev = data.get("operating_leverage", {})
                if op_lev:
                    ol_lines = []
                    dol = op_lev.get("dol_estimate", "N/A")
                    fv_mix = op_lev.get("fixed_variable_mix", "N/A")
                    be_assess = op_lev.get("breakeven_assessment", "N/A")
                    scalability = op_lev.get("scalability", "N/A")
                    interp = op_lev.get("interpretation", "")
                    ol_lines.append(f"**DOL Estimate**: {dol}")
                    ol_lines.append(f"**Cost Mix**: {fv_mix}")
                    ol_lines.append(f"**Breakeven**: {be_assess}")
                    ol_lines.append(f"**Scalability**: {scalability}")
                    if interp:
                        ol_lines.append(f"\n{interp}")
                    sections["Operating Leverage"] = "\n".join(ol_lines)
                    key_metrics["operating_leverage_dol"] = dol
                    key_metrics["scalability"] = scalability

                # Financial ratios section
                if "financial_ratios" in data:
                    ratio_lines = []
                    for ratio_name, ratio_data in data["financial_ratios"].items():
                        if isinstance(ratio_data, dict):
                            val = ratio_data.get("value", "N/A")
                            assessment = ratio_data.get("assessment", "")
                            ratio_lines.append(f"- **{ratio_name}**: {val} ({assessment})")
                            if assessment == "critical":
                                risk_flags.append(f"[FINANCIAL] {ratio_name}: {val} - CRITICAL")
                        else:
                            ratio_lines.append(f"- **{ratio_name}**: {ratio_data}")
                    if ratio_lines:
                        sections["Financial Ratios"] = "\n".join(ratio_lines)

                # Earnings Quality section
                eq = data.get("earnings_quality", {})
                if eq:
                    eq_lines = []
                    accrual = eq.get("accrual_ratio", {})
                    if accrual:
                        eq_lines.append(
                            f"**Accrual Ratio**: {accrual.get('value', 'N/A')} "
                            f"({accrual.get('assessment', '')})"
                        )
                    ocf_ni = eq.get("ocf_to_net_income", {})
                    if ocf_ni:
                        eq_lines.append(
                            f"**OCF/Net Income**: {ocf_ni.get('value', 'N/A')} "
                            f"({ocf_ni.get('assessment', '')})"
                        )
                    rev_flag = eq.get("revenue_recognition_flag", "")
                    if rev_flag:
                        eq_lines.append(f"**Revenue Recognition**: {rev_flag}")
                        if rev_flag == "Aggressive":
                            risk_flags.append("[FINANCIAL] Aggressive revenue recognition practices")
                    one_time = eq.get("one_time_items", "")
                    if one_time:
                        eq_lines.append(f"**One-time Items**: {one_time}")
                    if eq_lines:
                        sections["Earnings Quality"] = "\n".join(eq_lines)
                        key_metrics["earnings_quality"] = accrual.get("assessment", "N/A") if accrual else "N/A"

                # Financial sustainability
                fs = data.get("financial_sustainability", {})
                if fs:
                    fs_lines = []
                    fs_lines.append(f"**Cash Runway**: {fs.get('cash_runway_months', 'N/A')}")
                    fs_lines.append(f"**Can Fund Strategy**: {fs.get('can_fund_strategy', 'N/A')}")
                    fs_lines.append(f"**Key Constraint**: {fs.get('key_constraint', 'N/A')}")
                    fcf_gen = fs.get("fcf_generation", "")
                    if fcf_gen:
                        fs_lines.append(f"**FCF Generation**: {fcf_gen}")
                    sections["Financial Sustainability"] = "\n".join(fs_lines)
                    if fs.get("can_fund_strategy") == "No":
                        risk_flags.append("[FINANCIAL] Cannot fund current strategy with available resources")
                    if fs.get("source"):
                        citations.append(fs["source"])

                # Material changes section
                if "material_changes" in data and data["material_changes"]:
                    change_lines = []
                    for change in data["material_changes"]:
                        if isinstance(change, dict):
                            metric = change.get("metric", "")
                            chg = change.get("change", "")
                            sig = change.get("significance", "")
                            change_lines.append(f"- **{metric}**: {chg} - {sig}")
                            if change.get("source"):
                                citations.append(change["source"])
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
            skip_keys = {"financial_health_score", "roe_primary_driver", "earnings_quality",
                         "altman_zone", "scalability", "operating_leverage_dol"}
            for k, v in key_metrics.items():
                if k not in skip_keys:
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
            risk_flags=risk_flags,
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
