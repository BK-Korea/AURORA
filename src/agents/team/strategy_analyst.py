"""Strategy Analyst Agent - McKinsey-grade business strategy analysis from SEC filings."""
import json
import logging
from typing import Dict, Any

from .base_analyst import BaseAnalyst, AnalysisResult

logger = logging.getLogger(__name__)


class StrategyAnalyst(BaseAnalyst):
    """
    Specialist agent for McKinsey-grade strategic business analysis.

    Analyzes SEC filings from a strategic consulting perspective:
    - Business model viability and competitive positioning
    - Strategic spending assessment (R&D, SG&A, CapEx justification)
    - Revenue concentration and diversification risks
    - Management execution assessment vs stated strategy
    - Market opportunity sizing and penetration analysis
    - Operational efficiency and scalability
    """

    @property
    def analyst_type(self) -> str:
        return "strategy_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a **McKinsey Senior Partner** specializing in corporate strategy.
You analyze SEC filings to assess business viability, strategic positioning, and
management execution with the rigor expected at a top-tier strategy consulting firm.

## Strategic Analysis Framework

### 1. Business Model Assessment
- What is the company's core value proposition?
- Revenue model: recurring vs one-time, B2B vs B2C, product vs service
- Revenue concentration risk: customer, geographic, product dependency
- Moat/competitive advantage: IP, network effects, switching costs, scale

### 2. Strategic Spending Analysis (CRITICAL)
Evaluate whether spending levels are strategically justified:
- **R&D Spend**: Is the R&D investment proportional to market opportunity?
  Compare R&D-to-revenue ratio vs industry peers. Is it building future moats?
- **SG&A Spend**: Is sales cost justified by revenue growth? Customer acquisition cost trends
- **CapEx**: Is capital investment aligned with strategic priorities?
- **Cash Burn Analysis**: At current burn rate, what is the cash runway?
  Is the burn rate justified by growth stage and opportunity size?

### 3. Strategic Execution Assessment
- Is management delivering on stated strategic goals?
- Gap analysis: promised vs delivered milestones
- Resource allocation alignment with strategic priorities
- Organizational capability to execute the strategy

### 4. Competitive Positioning
- Market position and share (if disclosed)
- Competitive threats mentioned in risk factors
- Barriers to entry for competitors
- Strategic partnerships and ecosystem value

### 5. Growth Strategy Viability
- Is the growth strategy financially sustainable?
- Total addressable market (TAM) credibility
- Path to profitability analysis
- Unit economics assessment (if available)

## Output Format (JSON)
```json
{
  "business_model": {
    "core_business": "Description of core business",
    "revenue_model": "Recurring/One-time/Hybrid",
    "competitive_moat": "Description of competitive advantages",
    "moat_strength": "Strong/Moderate/Weak",
    "source": "[citation]"
  },
  "strategic_spending": {
    "rd_analysis": {
      "amount": "$X",
      "revenue_ratio": "X%",
      "assessment": "Justified/Excessive/Insufficient",
      "reasoning": "Why this level is or isn't justified",
      "source": "[citation]"
    },
    "sga_analysis": {
      "amount": "$X",
      "revenue_ratio": "X%",
      "assessment": "Efficient/Concerning/Bloated",
      "reasoning": "Analysis of SG&A efficiency",
      "source": "[citation]"
    },
    "capex_analysis": {
      "amount": "$X",
      "strategic_alignment": "Aligned/Misaligned",
      "reasoning": "How CapEx supports strategy",
      "source": "[citation]"
    },
    "cash_runway": {
      "months": "X months",
      "burn_rate": "$X/quarter",
      "sustainability": "Sustainable/At Risk/Critical",
      "source": "[citation]"
    }
  },
  "execution_assessment": {
    "milestones_delivered": ["milestone1", "milestone2"],
    "milestones_missed": ["milestone1"],
    "execution_score": 7,
    "source": "[citation]"
  },
  "competitive_position": {
    "market_position": "Leader/Challenger/Niche/Follower",
    "key_competitors": ["competitor1", "competitor2"],
    "differentiation": "Key differentiators",
    "threats": ["threat1", "threat2"],
    "source": "[citation]"
  },
  "growth_viability": {
    "tam_assessment": "TAM size and credibility",
    "path_to_profitability": "Clear/Unclear/Not applicable",
    "growth_sustainability": "Sustainable/Questionable/Unsustainable",
    "reasoning": "Detailed reasoning",
    "source": "[citation]"
  },
  "strategic_verdict": "Overall strategic assessment in 2-3 sentences",
  "strategy_score": 7,
  "strategy_narrative": "3-4 paragraph Korean narrative with McKinsey-style strategic insight"
}
```

## Rules
- Think like a McKinsey consultant: framework-driven, hypothesis-led, data-backed
- ALWAYS connect financial data to strategic implications
- Spending is not inherently bad - evaluate if it's STRATEGICALLY JUSTIFIED
- Compare ratios to industry norms when possible
- Provide actionable strategic recommendations
- Every claim must cite specific document source
- Include English original quotes for key strategic statements
- Provide narrative in Korean with professional consulting tone"""

    def analyze(self, question: str, context: str, metadata: Dict[str, Any]) -> AnalysisResult:
        self._report("Strategy Analyst: Analyzing business model and strategic positioning...")

        company_name = metadata.get("company_name", "Unknown")

        user_prompt = f"""## SEC Document Context

{context}

---

## Strategic Analysis Request

Company: {company_name}
Question: {question}

Perform a comprehensive McKinsey-grade strategic analysis.
Focus on: business model viability, spending justification, competitive positioning,
management execution, and growth sustainability.
Respond in the specified JSON format."""

        raw = self._invoke_llm(self.system_prompt, user_prompt)

        if not raw:
            return AnalysisResult(
                analyst_type=self.analyst_type,
                sections={},
                confidence=0.0,
                error="Strategy analysis LLM call failed",
            )

        return self._parse_result(raw)

    def _parse_result(self, raw: str) -> AnalysisResult:
        sections = {}
        key_metrics = {}
        risk_flags = []
        citations = []
        confidence = 0.7

        try:
            json_str = _extract_json(raw)
            if json_str:
                data = json.loads(json_str)

                # Business Model
                bm = data.get("business_model", {})
                if bm:
                    bm_text = (
                        f"**Core Business**: {bm.get('core_business', 'N/A')}\n"
                        f"**Revenue Model**: {bm.get('revenue_model', 'N/A')}\n"
                        f"**Competitive Moat**: {bm.get('competitive_moat', 'N/A')}\n"
                        f"**Moat Strength**: {bm.get('moat_strength', 'N/A')}"
                    )
                    sections["Business Model"] = bm_text
                    key_metrics["moat_strength"] = bm.get("moat_strength", "N/A")
                    src = bm.get("source", "")
                    if src:
                        citations.append(src)

                # Strategic Spending
                ss = data.get("strategic_spending", {})
                if ss:
                    spend_lines = []

                    rd = ss.get("rd_analysis", {})
                    if rd:
                        spend_lines.append(
                            f"**R&D**: {rd.get('amount', 'N/A')} "
                            f"({rd.get('revenue_ratio', 'N/A')} of revenue) - "
                            f"**{rd.get('assessment', 'N/A')}**\n"
                            f"  {rd.get('reasoning', '')}"
                        )
                        key_metrics["rd_to_revenue"] = rd.get("revenue_ratio", "N/A")
                        if rd.get("source"):
                            citations.append(rd["source"])

                    sga = ss.get("sga_analysis", {})
                    if sga:
                        spend_lines.append(
                            f"**SG&A**: {sga.get('amount', 'N/A')} "
                            f"({sga.get('revenue_ratio', 'N/A')} of revenue) - "
                            f"**{sga.get('assessment', 'N/A')}**\n"
                            f"  {sga.get('reasoning', '')}"
                        )
                        if sga.get("source"):
                            citations.append(sga["source"])

                    capex = ss.get("capex_analysis", {})
                    if capex:
                        spend_lines.append(
                            f"**CapEx**: {capex.get('amount', 'N/A')} - "
                            f"Strategic Alignment: **{capex.get('strategic_alignment', 'N/A')}**\n"
                            f"  {capex.get('reasoning', '')}"
                        )
                        if capex.get("source"):
                            citations.append(capex["source"])

                    runway = ss.get("cash_runway", {})
                    if runway:
                        sustainability = runway.get("sustainability", "N/A")
                        spend_lines.append(
                            f"**Cash Runway**: {runway.get('months', 'N/A')} "
                            f"(Burn: {runway.get('burn_rate', 'N/A')}) - "
                            f"**{sustainability}**"
                        )
                        key_metrics["cash_runway_months"] = runway.get("months", "N/A")
                        key_metrics["burn_sustainability"] = sustainability
                        if sustainability in ["At Risk", "Critical"]:
                            risk_flags.append(
                                f"[STRATEGY] Cash runway: {runway.get('months', '?')}, "
                                f"burn rate {runway.get('burn_rate', '?')}"
                            )
                        if runway.get("source"):
                            citations.append(runway["source"])

                    if spend_lines:
                        sections["Strategic Spending Assessment"] = "\n\n".join(spend_lines)

                # Execution Assessment
                exe = data.get("execution_assessment", {})
                if exe:
                    delivered = exe.get("milestones_delivered", [])
                    missed = exe.get("milestones_missed", [])
                    exe_score = exe.get("execution_score", 5)
                    exe_text = f"**Execution Score**: {exe_score}/10\n\n"
                    if delivered:
                        exe_text += "**Delivered**:\n" + "\n".join(
                            f"- {m}" for m in delivered
                        ) + "\n\n"
                    if missed:
                        exe_text += "**Missed/Delayed**:\n" + "\n".join(
                            f"- {m}" for m in missed
                        )
                        if len(missed) >= 2:
                            risk_flags.append(
                                f"[STRATEGY] {len(missed)} milestones missed/delayed"
                            )
                    sections["Management Execution"] = exe_text
                    key_metrics["execution_score"] = exe_score
                    if exe.get("source"):
                        citations.append(exe["source"])

                # Competitive Position
                cp = data.get("competitive_position", {})
                if cp:
                    cp_text = (
                        f"**Market Position**: {cp.get('market_position', 'N/A')}\n"
                        f"**Differentiation**: {cp.get('differentiation', 'N/A')}\n"
                    )
                    competitors = cp.get("key_competitors", [])
                    if competitors:
                        cp_text += f"**Key Competitors**: {', '.join(competitors)}\n"
                    threats = cp.get("threats", [])
                    if threats:
                        cp_text += "**Threats**:\n" + "\n".join(f"- {t}" for t in threats)
                    sections["Competitive Positioning"] = cp_text
                    key_metrics["market_position"] = cp.get("market_position", "N/A")
                    if cp.get("source"):
                        citations.append(cp["source"])

                # Growth Viability
                gv = data.get("growth_viability", {})
                if gv:
                    gv_text = (
                        f"**TAM**: {gv.get('tam_assessment', 'N/A')}\n"
                        f"**Path to Profitability**: {gv.get('path_to_profitability', 'N/A')}\n"
                        f"**Growth Sustainability**: {gv.get('growth_sustainability', 'N/A')}\n"
                        f"**Reasoning**: {gv.get('reasoning', 'N/A')}"
                    )
                    sections["Growth Strategy Viability"] = gv_text
                    sustainability = gv.get("growth_sustainability", "")
                    if sustainability in ["Questionable", "Unsustainable"]:
                        risk_flags.append(f"[STRATEGY] Growth sustainability: {sustainability}")
                    if gv.get("source"):
                        citations.append(gv["source"])

                # Scores
                strategy_score = data.get("strategy_score", 5)
                key_metrics["strategy_score"] = strategy_score
                confidence = min(1.0, strategy_score / 10.0)

                verdict = data.get("strategic_verdict", "")
                if verdict:
                    sections["Strategic Verdict"] = f"**{verdict}**"

                narrative = data.get("strategy_narrative", "")
                if narrative:
                    sections["Strategic Analysis"] = narrative

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse strategy analysis JSON: {e}")
            sections["Strategic Analysis"] = raw
            confidence = 0.5

        return AnalysisResult(
            analyst_type=self.analyst_type,
            sections=sections,
            confidence=confidence,
            key_metrics=key_metrics,
            risk_flags=risk_flags,
            citations=citations,
        )


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling code blocks."""
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
