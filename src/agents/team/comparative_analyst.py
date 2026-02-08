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
        return """You are a **Senior Equity Research Analyst** specializing in trend analysis.
Your role is to identify patterns, trends, and inflection points across multiple
SEC filing periods with the rigor expected at Goldman Sachs.

## Analysis Framework

### 1. Period-over-Period Comparison
For each available metric, compare across filing periods:
- Revenue trajectory
- Margin trends (expanding/contracting)
- Cash flow trajectory
- Debt level changes
- Operating expense trends
- Headcount changes (if available)

### 2. Trend Classification
Classify each trend as:
- **Accelerating**: Metric improving at increasing rate
- **Stable Growth**: Steady improvement
- **Decelerating**: Positive but slowing
- **Flat**: No significant change
- **Deteriorating**: Declining
- **Inflection**: Reversing direction

### 3. Management Guidance Assessment
- Extract any forward-looking statements
- Compare current results vs prior guidance
- Assess management credibility on projections

### 4. Key Inflection Points
Identify moments where the business trajectory changed materially.

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
      "significance": "High/Medium/Low",
      "analysis": "Brief interpretation"
    }
  ],
  "inflection_points": [
    {
      "event": "Description of inflection",
      "period": "Q2 2024",
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
  "trend_narrative": "2-3 paragraph Korean narrative for CEO briefing"
}
```

## Rules
- Compare at least 2 time periods wherever data is available
- Every data point MUST cite the exact filing source
- Include English original quotes for management guidance
- Use percentage changes for comparability
- If only single period data exists, note the limitation and analyze available trends
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
        citations = []
        confidence = 0.7

        try:
            json_str = self._extract_json(raw)
            if json_str:
                data = json.loads(json_str)

                # Parse trends
                trends = data.get("trends", [])
                if trends:
                    trend_table = [
                        "| Metric | Direction | Change | Significance |",
                        "|---|---|---|---|",
                    ]
                    trend_details = []

                    for trend in trends:
                        if not isinstance(trend, dict):
                            continue
                        metric = trend.get("metric", "")
                        direction = trend.get("trend_direction", "")
                        change = trend.get("change_pct", "N/A")
                        sig = trend.get("significance", "")
                        analysis = trend.get("analysis", "")

                        icon = self._trend_icon(direction)
                        trend_table.append(
                            f"| {metric} | {icon} {direction} | {change} | {sig} |"
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
                        if analysis:
                            detail += f"- {analysis}\n"
                        trend_details.append(detail)

                    sections["Trend Summary"] = "\n".join(trend_table)
                    if trend_details:
                        sections["Trend Details"] = "\n\n".join(trend_details)
                    key_metrics["trends_identified"] = len(trends)

                # Inflection points
                inflections = data.get("inflection_points", [])
                if inflections:
                    infl_lines = []
                    for infl in inflections:
                        if isinstance(infl, dict):
                            event = infl.get("event", "")
                            period = infl.get("period", "")
                            impact = infl.get("impact", "")
                            src = infl.get("source", "")
                            infl_lines.append(
                                f"- **{event}** ({period}): {impact} {src}"
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
