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
Your role is to systematically identify, categorize, and score ALL risks
disclosed in SEC filings with institutional-grade rigor.

## Risk Analysis Framework

### 1. Risk Identification
Scan ALL document sections for risk factors, especially:
- Item 1A (Risk Factors) in 10-K/10-Q
- Going concern language in financial statements
- Litigation disclosures
- Regulatory proceedings
- Management discussion of uncertainties

### 2. Risk Categorization
Categorize each risk into:
- **Financial Risk**: Liquidity, credit, currency, interest rate
- **Operational Risk**: Supply chain, technology failure, key personnel
- **Market Risk**: Competition, demand shifts, pricing pressure
- **Regulatory & Compliance Risk**: Government action, policy changes
- **Litigation Risk**: Pending lawsuits, investigations, settlements
- **Technology Risk**: Cybersecurity, IP, tech obsolescence
- **Going Concern Risk**: Ability to continue operations
- **Management & Governance Risk**: Leadership, board, internal controls

### 3. Risk Scoring (per risk)
- **Severity**: 1-5 (1=minor, 5=existential)
- **Probability**: 1-5 (1=unlikely, 5=near-certain)
- **Risk Score**: Severity x Probability (1-25)

### 4. Aggregate Risk Assessment
- **Overall Risk Rating**: LOW / MODERATE / ELEVATED / HIGH / CRITICAL
- **Risk Score**: Weighted average of individual risk scores (1-10)

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
      "source_quote": "Exact English quote from filing",
      "source_citation": "[Form Date | Page | Section]",
      "mitigation": "Any disclosed mitigation measures"
    }
  ],
  "going_concern_signals": [
    {"signal": "description", "source": "[citation]"}
  ],
  "overall_risk_rating": "MODERATE",
  "aggregate_risk_score": 6.5,
  "risk_summary_narrative": "2-3 paragraph Korean narrative for CEO briefing",
  "top_3_risks": ["risk1_title", "risk2_title", "risk3_title"]
}
```

## Rules
- Identify AT LEAST 5 distinct risks from the documents
- Every risk MUST cite specific document source
- Include English original quotes for critical risk language
- Going concern language gets automatic severity=5
- Be specific - "revenue decline risk" is too vague; "37% revenue concentration in single customer" is proper
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
                        "| Category | Risk | Severity | Prob. | Score |",
                        "|---|---|:---:|:---:|:---:|",
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
                        desc = risk.get("description", "")
                        quote = risk.get("source_quote", "")
                        citation = risk.get("source_citation", "")
                        mitigation = risk.get("mitigation", "")

                        risk_table_lines.append(
                            f"| {cat} | {title} | {sev}/5 | {prob}/5 | **{score}** |"
                        )

                        detail = f"#### {title}\n- **Category**: {cat}\n- **Score**: {score}/25\n"
                        if desc:
                            detail += f"- {desc}\n"
                        if quote:
                            detail += f'- Original: *"{quote}"*\n'
                        if citation:
                            detail += f"- Source: {citation}\n"
                            citations.append(citation)
                        if mitigation:
                            detail += f"- Mitigation: {mitigation}\n"
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
                key_metrics["overall_risk_rating"] = rating
                key_metrics["aggregate_risk_score"] = agg_score

                sections["Risk Rating"] = (
                    f"**Overall Risk Rating**: {rating}\n"
                    f"**Aggregate Risk Score**: {agg_score}/10"
                )

                # Confidence based on risk count and score
                if len(risks) >= 5:
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
