"""Risk Analyst Agent - Systematic risk identification and scoring from SEC filings."""
import json
import logging
from typing import Dict, Any, List

from .base_analyst import BaseAnalyst, AnalysisResult

logger = logging.getLogger(__name__)


class RiskAnalyst(BaseAnalyst):
    """
    Specialist agent for risk identification and quantitative risk scoring.

    Responsibilities:
    - Identify and categorize risks from SEC filings
    - Score each risk by severity and probability
    - Detect going concern signals
    - Flag regulatory and compliance risks
    - Assess litigation exposure
    """

    RISK_CATEGORIES = [
        "Financial Risk",
        "Operational Risk",
        "Market Risk",
        "Regulatory & Compliance Risk",
        "Litigation Risk",
        "Technology Risk",
        "Going Concern Risk",
        "Management & Governance Risk",
    ]

    @property
    def analyst_type(self) -> str:
        return "risk_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a **Chief Risk Officer** at a top-tier investment bank.
Your role is to systematically identify, categorize, score, and model ALL risks
disclosed in SEC filings with institutional-grade rigor. You apply advanced risk
management frameworks including scenario analysis and risk interdependency mapping.

## Risk Analysis Framework

### 1. Risk Identification
Scan ALL document sections for risk factors, especially:
- Item 1A (Risk Factors) in 10-K/10-Q
- Going concern language in financial statements
- Litigation disclosures
- Regulatory proceedings
- Management discussion of uncertainties
- Off-balance-sheet arrangements and contingent liabilities
- Related party transactions

### 2. Risk Categorization
Categorize each risk into:
- **Financial Risk**: Liquidity, credit, currency, interest rate, refinancing
- **Operational Risk**: Supply chain, technology failure, key personnel dependency
- **Market Risk**: Competition, demand shifts, pricing pressure, market cyclicality
- **Regulatory & Compliance Risk**: Government action, policy changes, sanctions
- **Litigation Risk**: Pending lawsuits, investigations, settlements, IP disputes
- **Technology Risk**: Cybersecurity, IP, tech obsolescence, AI disruption
- **Going Concern Risk**: Ability to continue operations
- **Management & Governance Risk**: Leadership, board independence, internal controls
- **ESG Risk**: Environmental liabilities, social responsibility, governance quality
- **Concentration Risk**: Customer, supplier, geographic, revenue concentration

### 3. Risk Scoring (per risk)
- **Severity**: 1-5 (1=minor, 5=existential)
- **Probability**: 1-5 (1=unlikely, 5=near-certain)
- **Risk Score**: Severity x Probability (1-25)
- **Velocity**: How quickly the risk could materialize (Immediate/Short-term/Medium-term/Long-term)
- **Trend**: Is this risk Increasing/Stable/Decreasing vs prior period?

### 4. Scenario Analysis (CRITICAL)
Develop three scenarios based on disclosed risks:
- **Bull Case**: Key risks do not materialize, positive catalysts play out
  - What financial outcomes result? Which metrics improve?
- **Base Case**: Most likely scenario given current trajectory
  - Expected financial performance range
- **Bear Case**: Multiple risks materialize simultaneously
  - Downside financial impact, cash burn acceleration, covenant breach risk
For each scenario, estimate probability (must sum to ~100%).

### 5. Risk Interdependency Analysis
Identify risk clusters where risks compound each other:
- Which risks are correlated (e.g., revenue decline + liquidity crisis)?
- Which risks could trigger cascading failures?
- What is the "worst-case cascade" - the domino chain of compounding risks?
- Identify the single "linchpin risk" whose materialization triggers the most others.

### 6. Concentration Risk Deep-Dive
Quantify all forms of concentration:
- **Customer concentration**: Top customer % of revenue, top 5 customers %
- **Geographic concentration**: Revenue by region
- **Product concentration**: Revenue by product/segment
- **Supplier concentration**: Single-source dependencies
- **Key person dependency**: Named individuals critical to operations

### 7. Aggregate Risk Assessment
- **Overall Risk Rating**: LOW / MODERATE / ELEVATED / HIGH / CRITICAL
- **Risk Score**: Weighted average of individual risk scores (1-10)
- **Risk Trajectory**: Is overall risk profile Improving/Stable/Deteriorating?

## Output Format (JSON)
```json
{
  "risks": [
    {
      "category": "Financial Risk",
      "title": "Short risk title",
      "description": "Detailed description with specific facts",
      "severity": 4,
      "probability": 3,
      "risk_score": 12,
      "velocity": "Short-term",
      "trend": "Increasing",
      "source_quote": "Exact English quote from filing",
      "source_citation": "[Form Date | Page | Section]",
      "mitigation": "Any disclosed mitigation measures",
      "connected_risks": ["titles of other risks this compounds with"]
    }
  ],
  "scenario_analysis": {
    "bull_case": {
      "probability": "25%",
      "description": "Detailed bull case scenario",
      "key_assumptions": ["assumption1", "assumption2"],
      "financial_impact": "Expected financial outcome"
    },
    "base_case": {
      "probability": "50%",
      "description": "Detailed base case scenario",
      "key_assumptions": ["assumption1", "assumption2"],
      "financial_impact": "Expected financial outcome"
    },
    "bear_case": {
      "probability": "25%",
      "description": "Detailed bear case scenario",
      "key_assumptions": ["assumption1", "assumption2"],
      "financial_impact": "Expected financial outcome"
    }
  },
  "risk_interdependencies": {
    "risk_clusters": [
      {"cluster_name": "name", "risks": ["risk1", "risk2"], "cascade_description": "How they compound"}
    ],
    "linchpin_risk": "The single risk that triggers the most cascading effects",
    "worst_case_cascade": "Description of the worst-case domino chain"
  },
  "concentration_risks": {
    "customer": {"top_customer_pct": "X%", "top5_pct": "X%", "assessment": "High/Moderate/Low", "source": "[citation]"},
    "geographic": {"primary_region_pct": "X%", "assessment": "High/Moderate/Low", "source": "[citation]"},
    "product": {"primary_product_pct": "X%", "assessment": "High/Moderate/Low", "source": "[citation]"},
    "key_person": {"dependency_level": "High/Moderate/Low", "details": "specifics", "source": "[citation]"}
  },
  "going_concern_signals": [
    {"signal": "description", "source": "[citation]"}
  ],
  "overall_risk_rating": "MODERATE",
  "aggregate_risk_score": 6.5,
  "risk_trajectory": "Improving/Stable/Deteriorating",
  "risk_summary_narrative": "3-4 paragraph Korean narrative for CEO briefing. Must cover: 핵심 리스크 요약, 시나리오별 시사점, 리스크 상호연관성, 집중도 리스크, 최종 리스크 판단.",
  "top_3_risks": ["risk1_title", "risk2_title", "risk3_title"]
}
```

## Rules
- Identify AT LEAST 7 distinct risks from the documents
- Every risk MUST cite specific document source with English original quotes
- Going concern language gets automatic severity=5
- Be specific - "revenue decline risk" is too vague; "37% revenue concentration in single customer" is proper
- Scenario analysis must be grounded in disclosed facts, not speculation
- Risk interdependencies must identify at least 2 risk clusters
- Always assess concentration risk even if not explicitly in risk factors (derive from financials)
- Provide the narrative in Korean"""

    def analyze(self, question: str, context: str, metadata: Dict[str, Any]) -> AnalysisResult:
        self._report("Risk Analyst: Scanning for risk factors...")

        company_name = metadata.get("company_name", "Unknown")

        user_prompt = f"""## SEC Document Context

{context}

---

## Risk Analysis Request

Company: {company_name}
Question: {question}

Perform a comprehensive risk analysis of the above SEC documents.
Identify ALL disclosed risks, categorize and score them.
Pay special attention to going concern language and litigation exposure.
Respond in the specified JSON format."""

        raw = self._invoke_llm(self.system_prompt, user_prompt)

        if not raw:
            return AnalysisResult(
                analyst_type=self.analyst_type,
                sections={},
                confidence=0.0,
                error="Risk analysis LLM call failed",
            )

        return self._parse_result(raw)

    def _parse_result(self, raw: str) -> AnalysisResult:
        sections = {}
        key_metrics = {}
        risk_flags = []
        citations = []
        confidence = 0.7

        try:
            json_str = self._extract_json(raw)
            if json_str:
                data = json.loads(json_str)

                # Parse individual risks
                risks = data.get("risks", [])
                if risks:
                    risk_table_lines = [
                        "| Category | Risk | Sev. | Prob. | Score | Velocity | Trend |",
                        "|---|---|:---:|:---:|:---:|---|---|",
                    ]
                    risk_detail_lines = []

                    for risk in risks:
                        if not isinstance(risk, dict):
                            continue
                        cat = risk.get("category", "Unknown")
                        title = risk.get("title", "")
                        sev = risk.get("severity", 0)
                        prob = risk.get("probability", 0)
                        score = risk.get("risk_score", sev * prob)
                        velocity = risk.get("velocity", "")
                        trend = risk.get("trend", "")
                        desc = risk.get("description", "")
                        quote = risk.get("source_quote", "")
                        citation = risk.get("source_citation", "")
                        mitigation = risk.get("mitigation", "")
                        connected = risk.get("connected_risks", [])

                        risk_table_lines.append(
                            f"| {cat} | {title} | {sev}/5 | {prob}/5 | **{score}** | {velocity} | {trend} |"
                        )

                        detail = f"#### {title}\n- **Category**: {cat}\n- **Score**: {score}/25"
                        if velocity:
                            detail += f" | **Velocity**: {velocity}"
                        if trend:
                            detail += f" | **Trend**: {trend}"
                        detail += "\n"
                        if desc:
                            detail += f"- {desc}\n"
                        if quote:
                            detail += f'- Original: *"{quote}"*\n'
                        if citation:
                            detail += f"- Source: {citation}\n"
                            citations.append(citation)
                        if mitigation:
                            detail += f"- Mitigation: {mitigation}\n"
                        if connected:
                            detail += f"- Connected Risks: {', '.join(connected)}\n"
                        risk_detail_lines.append(detail)

                        # Flag high-severity risks
                        if sev >= 4 or score >= 16:
                            risk_flags.append(f"[HIGH] {title} (Score: {score}/25)")

                    sections["Risk Matrix"] = "\n".join(risk_table_lines)
                    sections["Risk Details"] = "\n\n".join(risk_detail_lines)
                    key_metrics["total_risks_identified"] = len(risks)
                    key_metrics["high_severity_risks"] = len(
                        [r for r in risks if isinstance(r, dict) and r.get("severity", 0) >= 4]
                    )

                # Scenario Analysis
                scenarios = data.get("scenario_analysis", {})
                if scenarios:
                    scenario_lines = []
                    for case_key, case_label in [("bull_case", "Bull Case"),
                                                  ("base_case", "Base Case"),
                                                  ("bear_case", "Bear Case")]:
                        case = scenarios.get(case_key, {})
                        if case:
                            prob = case.get("probability", "N/A")
                            desc = case.get("description", "")
                            assumptions = case.get("key_assumptions", [])
                            impact = case.get("financial_impact", "")
                            scenario_lines.append(f"#### {case_label} (Probability: {prob})")
                            if desc:
                                scenario_lines.append(f"{desc}")
                            if assumptions:
                                scenario_lines.append("**Key Assumptions**:")
                                for a in assumptions:
                                    scenario_lines.append(f"- {a}")
                            if impact:
                                scenario_lines.append(f"**Financial Impact**: {impact}")
                            scenario_lines.append("")
                    if scenario_lines:
                        sections["Scenario Analysis"] = "\n".join(scenario_lines)

                # Risk Interdependencies
                interdeps = data.get("risk_interdependencies", {})
                if interdeps:
                    inter_lines = []
                    clusters = interdeps.get("risk_clusters", [])
                    if clusters:
                        inter_lines.append("**Risk Clusters**:")
                        for cluster in clusters:
                            if isinstance(cluster, dict):
                                name = cluster.get("cluster_name", "")
                                risks_in = cluster.get("risks", [])
                                cascade = cluster.get("cascade_description", "")
                                inter_lines.append(f"- **{name}**: {', '.join(risks_in)}")
                                if cascade:
                                    inter_lines.append(f"  Cascade: {cascade}")
                        inter_lines.append("")
                    linchpin = interdeps.get("linchpin_risk", "")
                    if linchpin:
                        inter_lines.append(f"**Linchpin Risk**: {linchpin}")
                        key_metrics["linchpin_risk"] = linchpin
                    worst = interdeps.get("worst_case_cascade", "")
                    if worst:
                        inter_lines.append(f"**Worst-Case Cascade**: {worst}")
                    if inter_lines:
                        sections["Risk Interdependencies"] = "\n".join(inter_lines)

                # Concentration Risks
                conc = data.get("concentration_risks", {})
                if conc:
                    conc_table = [
                        "| Type | Key Metric | Assessment |",
                        "|---|---|---|",
                    ]
                    for conc_type, conc_data in conc.items():
                        if isinstance(conc_data, dict):
                            assessment = conc_data.get("assessment", "N/A")
                            # Get the first non-standard key as the metric
                            metric_val = ""
                            for k, v in conc_data.items():
                                if k not in ("assessment", "source", "details"):
                                    metric_val = f"{k}: {v}"
                                    break
                            conc_table.append(
                                f"| {conc_type.replace('_', ' ').title()} | {metric_val} | {assessment} |"
                            )
                            if assessment == "High":
                                risk_flags.append(f"[CONCENTRATION] {conc_type}: {metric_val}")
                            src = conc_data.get("source", "")
                            if src:
                                citations.append(src)
                    if len(conc_table) > 2:
                        sections["Concentration Risk"] = "\n".join(conc_table)

                # Going concern signals
                gc_signals = data.get("going_concern_signals", [])
                if gc_signals:
                    gc_lines = []
                    for sig in gc_signals:
                        if isinstance(sig, dict):
                            gc_lines.append(
                                f"- {sig.get('signal', '')} {sig.get('source', '')}"
                            )
                            risk_flags.append(f"[GOING CONCERN] {sig.get('signal', '')}")
                    sections["Going Concern Signals"] = "\n".join(gc_lines)

                # Overall rating
                rating = data.get("overall_risk_rating", "UNKNOWN")
                agg_score = data.get("aggregate_risk_score", 5.0)
                trajectory = data.get("risk_trajectory", "Stable")
                key_metrics["overall_risk_rating"] = rating
                key_metrics["aggregate_risk_score"] = agg_score
                key_metrics["risk_trajectory"] = trajectory

                sections["Risk Rating"] = (
                    f"**Overall Risk Rating**: {rating}\n"
                    f"**Aggregate Risk Score**: {agg_score}/10\n"
                    f"**Risk Trajectory**: {trajectory}"
                )

                # Confidence based on risk count
                if len(risks) >= 7:
                    confidence = 0.9
                elif len(risks) >= 5:
                    confidence = 0.85
                elif len(risks) >= 3:
                    confidence = 0.7

                # Top 3 risks
                top3 = data.get("top_3_risks", [])
                if top3:
                    key_metrics["top_3_risks"] = top3

                # Narrative
                narrative = data.get("risk_summary_narrative", "")
                if narrative:
                    sections["Risk Analysis Summary"] = narrative

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse risk analysis JSON: {e}")
            sections["Risk Analysis"] = raw
            confidence = 0.5

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
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
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
