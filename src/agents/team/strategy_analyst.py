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
You apply classic strategic frameworks systematically to every analysis.

## Strategic Analysis Framework

### 1. Business Model Assessment
- What is the company's core value proposition?
- Revenue model: recurring vs one-time, B2B vs B2C, product vs service
- Revenue concentration risk: customer, geographic, product dependency
- Moat/competitive advantage: IP, network effects, switching costs, scale
- **Value Chain Analysis**: Identify primary activities (inbound logistics, operations,
  outbound logistics, marketing & sales, service) and support activities (infrastructure,
  HR, technology development, procurement). Where does the company capture the most margin?
  What is the company's position in the industry value chain? Is it moving up or down?

### 2. Porter's 5 Forces Analysis (CRITICAL FRAMEWORK)
Assess each force based ONLY on data disclosed in the SEC filings:
- **Threat of New Entrants**: Barriers to entry (capital requirements, IP, regulatory,
  scale economies). What does the company disclose about competitive moats?
- **Bargaining Power of Suppliers**: Supplier concentration, switching costs,
  input cost trends disclosed in COGS/cost of revenue
- **Bargaining Power of Buyers**: Customer concentration (top customer % of revenue),
  contract terms, pricing power indicators
- **Threat of Substitutes**: Alternative products/services mentioned in risk factors,
  technology disruption risks
- **Industry Rivalry**: Named competitors, market share data, pricing pressure,
  competitive dynamics described in MD&A
Rate each force: Strong/Moderate/Weak with supporting evidence.

### 3. Strategic Spending Analysis (CRITICAL)
Evaluate whether spending levels are strategically justified:
- **R&D Spend**: R&D-to-revenue ratio vs industry context. Is it building future moats?
  Classify R&D stage: basic research / product development / maintenance
- **SG&A Spend**: Is sales cost justified by revenue growth? Customer acquisition cost trends.
  Is the company achieving sales leverage (SG&A growing slower than revenue)?
- **CapEx**: Is capital investment aligned with strategic priorities?
  Maintenance CapEx vs Growth CapEx distinction
- **Cash Burn Analysis**: At current burn rate, cash runway in months.
  Is the burn rate justified by growth stage and opportunity size?

### 4. BCG Matrix / Strategic Portfolio Positioning
Based on available segment data, classify business units/products:
- **Stars**: High growth, high market share (invest aggressively)
- **Cash Cows**: Low growth, high market share (harvest for cash)
- **Question Marks**: High growth, low market share (selective investment or divest)
- **Dogs**: Low growth, low market share (divest or restructure)
If segment data is not available, assess the company's overall position.

### 5. Strategic Execution Assessment
- Is management delivering on stated strategic goals?
- Gap analysis: promised vs delivered milestones
- Resource allocation alignment with strategic priorities
- Capital allocation quality: are reinvestments going to the right areas?
- Management's track record: consistency of strategy over time

### 6. Competitive Positioning & SWOT Integration
- Market position and share (if disclosed)
- Competitive threats mentioned in risk factors
- **Strengths**: Internal capabilities and advantages (from all filings)
- **Weaknesses**: Internal limitations and vulnerabilities
- **Opportunities**: External growth possibilities mentioned in MD&A
- **Threats**: External risks from risk factors section
- Barriers to entry for competitors
- Strategic partnerships and ecosystem value

### 7. Growth Strategy Viability
- Is the growth strategy financially sustainable?
- Total addressable market (TAM) credibility assessment
- Path to profitability analysis with specific milestones
- Unit economics assessment (LTV/CAC if available)
- Organic vs inorganic growth strategy evaluation

## Output Format (JSON)
```json
{
  "business_model": {
    "core_business": "Description of core business",
    "revenue_model": "Recurring/One-time/Hybrid",
    "competitive_moat": "Description of competitive advantages",
    "moat_strength": "Strong/Moderate/Weak",
    "value_chain_position": "Where in the value chain and margin capture points",
    "source": "[citation]"
  },
  "porters_five_forces": {
    "new_entrants": {"strength": "Strong/Moderate/Weak", "evidence": "specific evidence", "source": "[citation]"},
    "supplier_power": {"strength": "Strong/Moderate/Weak", "evidence": "specific evidence", "source": "[citation]"},
    "buyer_power": {"strength": "Strong/Moderate/Weak", "evidence": "specific evidence", "source": "[citation]"},
    "substitutes": {"strength": "Strong/Moderate/Weak", "evidence": "specific evidence", "source": "[citation]"},
    "industry_rivalry": {"strength": "Strong/Moderate/Weak", "evidence": "specific evidence", "source": "[citation]"},
    "overall_industry_attractiveness": "Attractive/Neutral/Unattractive"
  },
  "strategic_spending": {
    "rd_analysis": {
      "amount": "$X",
      "revenue_ratio": "X%",
      "rd_stage": "Basic Research/Product Development/Maintenance",
      "assessment": "Justified/Excessive/Insufficient",
      "reasoning": "Why this level is or isn't justified",
      "source": "[citation]"
    },
    "sga_analysis": {
      "amount": "$X",
      "revenue_ratio": "X%",
      "sales_leverage": "Achieving/Not Achieving",
      "assessment": "Efficient/Concerning/Bloated",
      "reasoning": "Analysis of SG&A efficiency",
      "source": "[citation]"
    },
    "capex_analysis": {
      "amount": "$X",
      "growth_vs_maintenance": "Growth-oriented/Maintenance-focused/Mixed",
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
  "bcg_positioning": {
    "classification": "Star/Cash Cow/Question Mark/Dog",
    "growth_rate": "High/Low indicator with data",
    "market_position": "Strong/Weak indicator with data",
    "strategic_implication": "Invest/Harvest/Selective/Divest recommendation",
    "source": "[citation]"
  },
  "swot": {
    "strengths": ["strength1 with citation", "strength2 with citation"],
    "weaknesses": ["weakness1 with citation", "weakness2 with citation"],
    "opportunities": ["opportunity1 with citation", "opportunity2 with citation"],
    "threats": ["threat1 with citation", "threat2 with citation"]
  },
  "execution_assessment": {
    "milestones_delivered": ["milestone1", "milestone2"],
    "milestones_missed": ["milestone1"],
    "capital_allocation_quality": "Excellent/Good/Poor",
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
    "growth_type": "Organic/Inorganic/Mixed",
    "growth_sustainability": "Sustainable/Questionable/Unsustainable",
    "reasoning": "Detailed reasoning",
    "source": "[citation]"
  },
  "strategic_verdict": "Overall strategic assessment in 2-3 sentences",
  "strategy_score": 7,
  "strategy_narrative": "4-5 paragraph Korean narrative with McKinsey-style strategic insight. Must cover: Porter's 5 Forces 요약, 사업모델 경쟁력, 비용 전략적 타당성, SWOT 통합 시사점, 최종 전략 판단."
}
```

## Rules
- Think like a McKinsey consultant: framework-driven, hypothesis-led, data-backed
- ALWAYS apply Porter's 5 Forces and Value Chain analysis systematically
- ALWAYS connect financial data to strategic implications
- Spending is not inherently bad - evaluate if it's STRATEGICALLY JUSTIFIED
- Compare ratios to industry norms when possible
- SWOT must be grounded in filing data, not speculation
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

                # Business Model + Value Chain
                bm = data.get("business_model", {})
                if bm:
                    bm_text = (
                        f"**Core Business**: {bm.get('core_business', 'N/A')}\n"
                        f"**Revenue Model**: {bm.get('revenue_model', 'N/A')}\n"
                        f"**Competitive Moat**: {bm.get('competitive_moat', 'N/A')}\n"
                        f"**Moat Strength**: {bm.get('moat_strength', 'N/A')}"
                    )
                    vcp = bm.get("value_chain_position", "")
                    if vcp:
                        bm_text += f"\n**Value Chain Position**: {vcp}"
                    sections["Business Model & Value Chain"] = bm_text
                    key_metrics["moat_strength"] = bm.get("moat_strength", "N/A")
                    src = bm.get("source", "")
                    if src:
                        citations.append(src)

                # Porter's 5 Forces
                p5f = data.get("porters_five_forces", {})
                if p5f:
                    force_names = {
                        "new_entrants": "Threat of New Entrants",
                        "supplier_power": "Supplier Bargaining Power",
                        "buyer_power": "Buyer Bargaining Power",
                        "substitutes": "Threat of Substitutes",
                        "industry_rivalry": "Industry Rivalry",
                    }
                    p5f_lines = ["| Force | Strength | Evidence |", "|---|---|---|"]
                    for key, label in force_names.items():
                        force = p5f.get(key, {})
                        if isinstance(force, dict):
                            strength = force.get("strength", "N/A")
                            evidence = force.get("evidence", "")
                            p5f_lines.append(f"| {label} | **{strength}** | {evidence} |")
                            src = force.get("source", "")
                            if src:
                                citations.append(src)
                    attractiveness = p5f.get("overall_industry_attractiveness", "N/A")
                    p5f_lines.append(f"\n**Overall Industry Attractiveness**: {attractiveness}")
                    sections["Porter's 5 Forces"] = "\n".join(p5f_lines)
                    key_metrics["industry_attractiveness"] = attractiveness

                # Strategic Spending
                ss = data.get("strategic_spending", {})
                if ss:
                    spend_lines = []

                    rd = ss.get("rd_analysis", {})
                    if rd:
                        rd_stage = rd.get("rd_stage", "")
                        stage_str = f" [{rd_stage}]" if rd_stage else ""
                        spend_lines.append(
                            f"**R&D**: {rd.get('amount', 'N/A')} "
                            f"({rd.get('revenue_ratio', 'N/A')} of revenue){stage_str} - "
                            f"**{rd.get('assessment', 'N/A')}**\n"
                            f"  {rd.get('reasoning', '')}"
                        )
                        key_metrics["rd_to_revenue"] = rd.get("revenue_ratio", "N/A")
                        if rd.get("source"):
                            citations.append(rd["source"])

                    sga = ss.get("sga_analysis", {})
                    if sga:
                        leverage = sga.get("sales_leverage", "")
                        lev_str = f" [Sales Leverage: {leverage}]" if leverage else ""
                        spend_lines.append(
                            f"**SG&A**: {sga.get('amount', 'N/A')} "
                            f"({sga.get('revenue_ratio', 'N/A')} of revenue){lev_str} - "
                            f"**{sga.get('assessment', 'N/A')}**\n"
                            f"  {sga.get('reasoning', '')}"
                        )
                        if sga.get("source"):
                            citations.append(sga["source"])

                    capex = ss.get("capex_analysis", {})
                    if capex:
                        gvm = capex.get("growth_vs_maintenance", "")
                        gvm_str = f" [{gvm}]" if gvm else ""
                        spend_lines.append(
                            f"**CapEx**: {capex.get('amount', 'N/A')}{gvm_str} - "
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

                # BCG Matrix Positioning
                bcg = data.get("bcg_positioning", {})
                if bcg:
                    bcg_text = (
                        f"**Classification**: {bcg.get('classification', 'N/A')}\n"
                        f"**Growth Rate**: {bcg.get('growth_rate', 'N/A')}\n"
                        f"**Market Position**: {bcg.get('market_position', 'N/A')}\n"
                        f"**Strategic Implication**: {bcg.get('strategic_implication', 'N/A')}"
                    )
                    sections["BCG Matrix Positioning"] = bcg_text
                    key_metrics["bcg_classification"] = bcg.get("classification", "N/A")
                    if bcg.get("source"):
                        citations.append(bcg["source"])

                # SWOT Analysis
                swot = data.get("swot", {})
                if swot:
                    swot_lines = []
                    for label, key in [("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                                       ("Opportunities", "opportunities"), ("Threats", "threats")]:
                        items = swot.get(key, [])
                        if items:
                            swot_lines.append(f"**{label}**:")
                            for item in items:
                                swot_lines.append(f"- {item}")
                            swot_lines.append("")
                    if swot_lines:
                        sections["SWOT Analysis"] = "\n".join(swot_lines)

                # Execution Assessment
                exe = data.get("execution_assessment", {})
                if exe:
                    delivered = exe.get("milestones_delivered", [])
                    missed = exe.get("milestones_missed", [])
                    exe_score = exe.get("execution_score", 5)
                    cap_alloc = exe.get("capital_allocation_quality", "")
                    exe_text = f"**Execution Score**: {exe_score}/10\n"
                    if cap_alloc:
                        exe_text += f"**Capital Allocation Quality**: {cap_alloc}\n"
                    exe_text += "\n"
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
                    if cap_alloc:
                        key_metrics["capital_allocation_quality"] = cap_alloc
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
                        f"**Growth Type**: {gv.get('growth_type', 'N/A')}\n"
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
