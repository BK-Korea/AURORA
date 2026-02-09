"""Comparative Analyst Agent - Cross-period trend analysis and peer context."""
import json
import logging
from typing import Dict, Any, List

from .base_analyst import BaseAnalyst, AnalysisResult

logger = logging.getLogger(__name__)


class ComparativeAnalyst(BaseAnalyst):
    """
    Specialist agent for temporal comparison and trend detection.

    Responsibilities:
    - Compare financial metrics across filing periods
    - Detect positive/negative trends
    - Identify inflection points and trajectory changes
    - Provide forward-looking context from management guidance
    - Assess execution against prior commitments
    """

    @property
    def analyst_type(self) -> str:
        return "comparative_analyst"

    @property
    def system_prompt(self) -> str:
        return """You are a **Senior Equity Research Analyst** specializing in trend analysis
and cross-period intelligence at Goldman Sachs. You apply advanced comparative
frameworks including management credibility scoring, earnings quality tracking,
and regression-to-mean analysis.

## Analysis Framework

### 1. Period-over-Period Comparison
For each available metric, compare across filing periods:
- Revenue trajectory (total and per-segment if available)
- Margin trends (gross, operating, net - expanding/contracting)
- Cash flow trajectory (operating, free cash flow)
- Debt level changes and leverage ratio trends
- Operating expense trends (R&D, SG&A, COGS individually)
- Headcount changes (if available)
- Working capital efficiency trends

### 2. Trend Classification
Classify each trend as:
- **Accelerating**: Metric improving at increasing rate
- **Stable Growth**: Steady improvement
- **Decelerating**: Positive but slowing
- **Flat**: No significant change
- **Deteriorating**: Declining
- **Inflection**: Reversing direction
For each trend, calculate the **rate of change of the rate of change** (2nd derivative)
to distinguish true acceleration from constant growth.

### 3. Management Credibility Score (CRITICAL FRAMEWORK)
Build a systematic credibility assessment:
- Extract ALL forward-looking statements from prior filings
- Compare each to actual delivered results
- Score: **Credibility Score = (Promises Met or Exceeded) / (Total Promises Made) x 100**
- Classify management as: **High Credibility (>75%) / Moderate (50-75%) / Low (<50%)**
- Track consistency: Does management revise guidance frequently? In which direction?
- Identify any patterns of "sandbagging" (low guidance + beat) or "over-promising"

### 4. Earnings Quality Trend Analysis
Track these quality indicators across periods:
- **Accrual Ratio Trend**: Is earnings quality improving or deteriorating?
- **Cash Conversion Trend**: OCF/Net Income ratio over time
- **Revenue Quality**: Recurring vs one-time revenue proportion changes
- **Expense Timing**: Any signs of expense deferral or acceleration
- **Working Capital Changes**: Unusual inventory or receivables build-up

### 5. Regression-to-Mean Analysis
For abnormally high or low metrics, assess:
- Is the current performance sustainable or likely to revert to historical average?
- What is the company's normalized earnings/margin level?
- Are current margins unsustainably high (competition will erode) or
  unsustainably low (temporary headwinds)?

### 6. Seasonality & Cyclicality Detection
- Identify quarterly seasonal patterns in revenue, margins, cash flow
- Detect industry cyclicality effects on performance
- Separate secular trends from cyclical fluctuations

### 7. Key Inflection Points
Identify moments where the business trajectory changed materially.
For each, assess whether the inflection is structural or temporary.

## Output Format (JSON)
```json
{
  "trends": [
    {
      "metric": "Revenue",
      "periods": [
        {"period": "Q1 2024", "value": "$X", "source": "[citation]"},
        {"period": "Q2 2024", "value": "$X", "source": "[citation]"}
      ],
      "trend_direction": "Accelerating",
      "change_pct": "+15%",
      "second_derivative": "Accelerating/Constant/Decelerating",
      "significance": "High/Medium/Low",
      "mean_reversion_risk": "Low/Moderate/High",
      "analysis": "Brief interpretation"
    }
  ],
  "management_credibility": {
    "promises_tracked": [
      {
        "promise": "What management said they would do",
        "promise_period": "FY2023",
        "result": "What actually happened",
        "result_period": "FY2024",
        "verdict": "Met/Exceeded/Missed/Partially Met",
        "source": "[citation]"
      }
    ],
    "credibility_score": "X%",
    "credibility_rating": "High/Moderate/Low",
    "guidance_pattern": "Conservative (sandbagging) / Accurate / Over-promising",
    "guidance_revision_frequency": "Rare/Occasional/Frequent"
  },
  "earnings_quality_trend": {
    "accrual_ratio_trend": "Improving/Stable/Deteriorating",
    "cash_conversion_trend": "Improving/Stable/Deteriorating",
    "revenue_quality_trend": "Improving/Stable/Deteriorating",
    "key_concerns": ["concern1", "concern2"],
    "overall_quality_trajectory": "Improving/Stable/Deteriorating"
  },
  "seasonality": {
    "detected": true,
    "pattern": "Description of seasonal patterns observed",
    "strongest_quarter": "Q4",
    "weakest_quarter": "Q1",
    "cyclicality_exposure": "High/Moderate/Low"
  },
  "inflection_points": [
    {
      "event": "Description of inflection",
      "period": "Q2 2024",
      "type": "Structural/Temporary",
      "impact": "Description of impact",
      "source": "[citation]"
    }
  ],
  "management_guidance": [
    {
      "statement": "Forward-looking statement",
      "period": "FY2024",
      "assessment": "On track / Behind / Exceeded",
      "source": "[citation]"
    }
  ],
  "overall_trajectory": "Improving / Stable / Deteriorating",
  "trajectory_confidence": 0.8,
  "trend_narrative": "3-4 paragraph Korean narrative for CEO briefing. Must cover: 핵심 트렌드 요약, 경영진 신뢰도 평가, 이익의 질 변화, 평균회귀 리스크, 최종 궤적 판단."
}
```

## Rules
- Compare at least 2 time periods wherever data is available
- Every data point MUST cite the exact filing source
- Include English original quotes for management guidance
- Use percentage changes for comparability
- Management credibility must be scored systematically with specific promises tracked
- Flag any earnings quality deterioration trends prominently
- If only single period data exists, note the limitation and analyze available trends
- Distinguish between secular trends and cyclical/seasonal effects
- Provide narrative in Korean"""

    def analyze(self, question: str, context: str, metadata: Dict[str, Any]) -> AnalysisResult:
        self._report("Comparative Analyst: Analyzing cross-period trends...")

        company_name = metadata.get("company_name", "Unknown")

        user_prompt = f"""## SEC Document Context

{context}

---

## Trend Analysis Request

Company: {company_name}
Question: {question}

Analyze trends and patterns across all available filing periods in the documents above.
Compare metrics period-over-period and identify inflection points.
Respond in the specified JSON format."""

        raw = self._invoke_llm(self.system_prompt, user_prompt)

        if not raw:
            return AnalysisResult(
                analyst_type=self.analyst_type,
                sections={},
                confidence=0.0,
                error="Comparative analysis LLM call failed",
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

                # Parse trends (enhanced with 2nd derivative and mean reversion)
                trends = data.get("trends", [])
                if trends:
                    trend_table = [
                        "| Metric | Direction | Change | 2nd Deriv. | Mean Reversion | Sig. |",
                        "|---|---|---|---|---|---|",
                    ]
                    trend_details = []

                    for trend in trends:
                        if not isinstance(trend, dict):
                            continue
                        metric = trend.get("metric", "")
                        direction = trend.get("trend_direction", "")
                        change = trend.get("change_pct", "N/A")
                        second_d = trend.get("second_derivative", "")
                        mean_rev = trend.get("mean_reversion_risk", "")
                        sig = trend.get("significance", "")
                        analysis = trend.get("analysis", "")

                        icon = self._trend_icon(direction)
                        trend_table.append(
                            f"| {metric} | {icon} {direction} | {change} | {second_d} | {mean_rev} | {sig} |"
                        )

                        # Collect period data
                        periods = trend.get("periods", [])
                        period_str = ""
                        if periods:
                            period_parts = []
                            for p in periods:
                                if isinstance(p, dict):
                                    period_parts.append(
                                        f"{p.get('period', '')}: {p.get('value', '')}"
                                    )
                                    src = p.get("source", "")
                                    if src:
                                        citations.append(src)
                            period_str = " -> ".join(period_parts)

                        detail = f"**{metric}** ({direction})\n"
                        if period_str:
                            detail += f"- Periods: {period_str}\n"
                        if second_d:
                            detail += f"- Momentum: {second_d}\n"
                        if mean_rev and mean_rev in ["Moderate", "High"]:
                            detail += f"- Mean Reversion Risk: **{mean_rev}**\n"
                            if mean_rev == "High":
                                risk_flags.append(f"[TREND] {metric}: High mean reversion risk")
                        if analysis:
                            detail += f"- {analysis}\n"
                        trend_details.append(detail)

                    sections["Trend Summary"] = "\n".join(trend_table)
                    if trend_details:
                        sections["Trend Details"] = "\n\n".join(trend_details)
                    key_metrics["trends_identified"] = len(trends)

                # Management Credibility Score
                mc = data.get("management_credibility", {})
                if mc:
                    promises = mc.get("promises_tracked", [])
                    cred_score = mc.get("credibility_score", "N/A")
                    cred_rating = mc.get("credibility_rating", "N/A")
                    guidance_pattern = mc.get("guidance_pattern", "")
                    revision_freq = mc.get("guidance_revision_frequency", "")

                    mc_lines = [
                        f"**Credibility Score**: {cred_score} ({cred_rating})",
                        f"**Guidance Pattern**: {guidance_pattern}",
                        f"**Revision Frequency**: {revision_freq}",
                    ]

                    if promises:
                        mc_lines.append("\n| Promise | Period | Result | Verdict |")
                        mc_lines.append("|---|---|---|---|")
                        for p in promises:
                            if isinstance(p, dict):
                                promise = p.get("promise", "")
                                p_period = p.get("promise_period", "")
                                result = p.get("result", "")
                                verdict = p.get("verdict", "")
                                mc_lines.append(f"| {promise} | {p_period} | {result} | **{verdict}** |")
                                src = p.get("source", "")
                                if src:
                                    citations.append(src)

                    sections["Management Credibility"] = "\n".join(mc_lines)
                    key_metrics["management_credibility_score"] = cred_score
                    key_metrics["management_credibility_rating"] = cred_rating
                    if cred_rating == "Low":
                        risk_flags.append(f"[CREDIBILITY] Management credibility rated Low: {cred_score}")

                # Earnings Quality Trend
                eqt = data.get("earnings_quality_trend", {})
                if eqt:
                    eq_lines = []
                    for metric_key, label in [
                        ("accrual_ratio_trend", "Accrual Ratio"),
                        ("cash_conversion_trend", "Cash Conversion"),
                        ("revenue_quality_trend", "Revenue Quality"),
                    ]:
                        val = eqt.get(metric_key, "")
                        if val:
                            eq_lines.append(f"- **{label}**: {val}")
                    concerns = eqt.get("key_concerns", [])
                    if concerns:
                        eq_lines.append("\n**Key Concerns**:")
                        for c in concerns:
                            eq_lines.append(f"- {c}")
                    overall_eq = eqt.get("overall_quality_trajectory", "")
                    if overall_eq:
                        eq_lines.append(f"\n**Overall Quality Trajectory**: {overall_eq}")
                        key_metrics["earnings_quality_trajectory"] = overall_eq
                        if overall_eq == "Deteriorating":
                            risk_flags.append("[QUALITY] Earnings quality deteriorating over time")
                    if eq_lines:
                        sections["Earnings Quality Trend"] = "\n".join(eq_lines)

                # Seasonality
                season = data.get("seasonality", {})
                if season and season.get("detected"):
                    s_lines = []
                    pattern = season.get("pattern", "")
                    if pattern:
                        s_lines.append(f"**Pattern**: {pattern}")
                    strongest = season.get("strongest_quarter", "")
                    weakest = season.get("weakest_quarter", "")
                    if strongest:
                        s_lines.append(f"**Strongest Period**: {strongest}")
                    if weakest:
                        s_lines.append(f"**Weakest Period**: {weakest}")
                    cyclicality = season.get("cyclicality_exposure", "")
                    if cyclicality:
                        s_lines.append(f"**Cyclicality Exposure**: {cyclicality}")
                    if s_lines:
                        sections["Seasonality & Cyclicality"] = "\n".join(s_lines)

                # Inflection points (enhanced with type)
                inflections = data.get("inflection_points", [])
                if inflections:
                    infl_lines = []
                    for infl in inflections:
                        if isinstance(infl, dict):
                            event = infl.get("event", "")
                            period = infl.get("period", "")
                            infl_type = infl.get("type", "")
                            impact = infl.get("impact", "")
                            src = infl.get("source", "")
                            type_str = f" [{infl_type}]" if infl_type else ""
                            infl_lines.append(
                                f"- **{event}** ({period}){type_str}: {impact} {src}"
                            )
                            if src:
                                citations.append(src)
                    sections["Inflection Points"] = "\n".join(infl_lines)

                # Management guidance
                guidance = data.get("management_guidance", [])
                if guidance:
                    guid_lines = []
                    for g in guidance:
                        if isinstance(g, dict):
                            stmt = g.get("statement", "")
                            assessment = g.get("assessment", "")
                            src = g.get("source", "")
                            guid_lines.append(
                                f'- *"{stmt}"* - **{assessment}** {src}'
                            )
                            if src:
                                citations.append(src)
                    sections["Management Guidance"] = "\n".join(guid_lines)

                # Overall trajectory
                trajectory = data.get("overall_trajectory", "Unknown")
                traj_conf = data.get("trajectory_confidence", 0.7)
                key_metrics["overall_trajectory"] = trajectory
                key_metrics["trajectory_confidence"] = traj_conf
                confidence = traj_conf

                sections["Trajectory Assessment"] = (
                    f"**Overall Trajectory**: {trajectory}\n"
                    f"**Confidence**: {traj_conf:.0%}"
                )

                # Narrative
                narrative = data.get("trend_narrative", "")
                if narrative:
                    sections["Comparative Analysis"] = narrative

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse comparative analysis JSON: {e}")
            sections["Comparative Analysis"] = raw
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
    def _trend_icon(direction: str) -> str:
        icons = {
            "Accelerating": "[++]",
            "Stable Growth": "[+]",
            "Decelerating": "[~+]",
            "Flat": "[=]",
            "Deteriorating": "[-]",
            "Inflection": "[!]",
        }
        return icons.get(direction, "[?]")

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
