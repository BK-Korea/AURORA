"""Report Synthesizer - Combines all analyst outputs into an executive-grade report."""
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base_analyst import AnalysisResult

logger = logging.getLogger(__name__)


class ReportSynthesizer:
    """
    Synthesizes outputs from all specialist analysts into a unified
    Goldman Sachs-grade executive research report.

    Takes structured AnalysisResults from:
    - Financial Analyst
    - Risk Analyst
    - Comparative Analyst

    Produces a single, cohesive report with:
    - Executive Summary (1-page brief)
    - Investment Highlights / Key Findings
    - Detailed Analysis sections
    - Risk Dashboard
    - Forward Outlook
    """

    SYNTHESIS_PROMPT = """당신은 골드만삭스 리서치 부서의 **Managing Director**입니다.
3명의 전문 애널리스트(재무, 리스크, 비교분석)가 각각 분석한 결과를 받아서
CEO에게 보고할 **최종 통합 리서치 리포트**를 작성합니다.

## 리포트 구조 (Goldman Sachs + McKinsey 통합 관점)

### 1. EXECUTIVE SUMMARY
- 3-5문장으로 핵심 결론 (사업 + 재무 동시에)
- 가장 중요한 수치 2-3개 반드시 포함
- **전략적 판단**: 사업영위에 문제가 있는지 Yes/No 명확히
- 투자/사업 관점에서의 핵심 시사점

### 2. STRATEGIC-FINANCIAL DASHBOARD
- 핵심 재무 지표 + 전략 지표를 통합 테이블로 정리
- 각 지표의 상태 (Positive / Neutral / Negative) 표시
- 전략 점수와 재무 건전성 점수 모두 포함

### 3. BUSINESS STRATEGY ASSESSMENT (맥킨지 관점)
- 사업 모델과 경쟁 포지셔닝 분석
- 비용 전략적 타당성: R&D, SG&A, CapEx가 전략에 부합하는지
- 경영진 실행력 평가
- 성장 전략의 실현 가능성

### 4. FINANCIAL DEEP DIVE (골드만삭스 관점)
- 핵심 재무 지표와 비율 분석
- 비용 구조 분석: 매출 대비 비용 비율과 추세
- 재무 지속가능성: 현금 런웨이, 부채 구조
- **핵심**: 재무가 사업계획을 뒷받침하는지 판단

### 5. STRATEGY-FINANCE LINKAGE (핵심 통합 분석)
이것이 가장 중요한 섹션. 다음 질문에 반드시 답해야 함:
- 재무 데이터 기반으로 사업영위에 문제는 없는지?
- 비용이 많이 나가는데 전략적으로 타당한지?
- 사업계획이 재무적으로 뒷받침되는지?
- 현재의 투자/비용이 미래 수익으로 전환될 가능성은?

### 6. TREND & TRAJECTORY
- 기간별 비교 분석
- 핵심 트렌드와 변곡점
- 경영진 가이던스 대비 실적 평가

### 7. RISK ASSESSMENT
- 리스크 매트릭스 (재무 + 사업 + 전략 통합)
- 상위 3대 리스크 상세 분석
- Going Concern 시그널 (해당시)
- 전체 리스크 등급

### 8. RECOMMENDATIONS & MONITORING POINTS
- CEO/투자자 관점 실행 가능한 권고사항
- 전략적 모니터링 포인트
- 후속 심층 분석이 필요한 영역

## 작성 원칙
1. **전략+재무 통합**: 숫자만 나열하지 말고, 항상 전략적 의미를 연결
2. **데이터 우선**: 모든 주장에 구체적 수치와 출처
3. **원문 인용**: 핵심 문구는 영어 원문 직접 인용
4. **균형적 시각**: 긍정/부정 요소 모두 객관적으로 서술
5. **실행 가능**: CEO가 의사결정에 바로 활용 가능한 인사이트
6. **한국어 작성**: 전체 리포트는 한국어, 원문 인용만 영어
7. **비용 ≠ 나쁨**: 비용이 높다고 무조건 부정적이 아님. 전략적 맥락에서 평가"""

    def __init__(
        self,
        llm: BaseChatModel,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.llm = llm
        self.progress_callback = progress_callback

    def _report(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)

    def synthesize(
        self,
        question: str,
        analyst_results: List[AnalysisResult],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize all analyst results into a unified report.

        Args:
            question: Original user question
            analyst_results: List of AnalysisResult from each analyst
            metadata: Additional context (company info, etc.)

        Returns:
            Dict with 'report', 'risk_rating', 'confidence', 'key_metrics'
        """
        self._report("Report Synthesizer: Compiling executive report...")

        # Build analyst inputs summary
        analyst_inputs = self._format_analyst_inputs(analyst_results)

        # Aggregate key metrics
        all_metrics = {}
        all_risk_flags = []
        all_citations = []
        avg_confidence = 0.0

        for result in analyst_results:
            if result.error:
                continue
            all_metrics.update(result.key_metrics)
            all_risk_flags.extend(result.risk_flags)
            all_citations.extend(result.citations)
            avg_confidence += result.confidence

        if analyst_results:
            valid_results = [r for r in analyst_results if not r.error]
            if valid_results:
                avg_confidence /= len(valid_results)

        company_name = metadata.get("company_name", "Unknown")

        user_prompt = f"""## 분석 대상

Company: {company_name}
Original Question: {question}

---

## 전문 애널리스트 분석 결과

{analyst_inputs}

---

위 3명의 애널리스트 분석 결과를 종합하여 **최종 CEO 브리핑 리포트**를 작성하세요.
모든 수치와 인용은 애널리스트 분석에서 가져오세요. 새로운 정보를 추가하지 마세요."""

        messages = [
            SystemMessage(content=self.SYNTHESIS_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            report = response.content
        except Exception as e:
            logger.error(f"Report synthesis failed: {e}")
            # Fallback: concatenate analyst sections
            report = self._fallback_report(analyst_results, question, company_name)

        self._report("Report Synthesizer: Executive report complete.")

        return {
            "report": report,
            "risk_rating": all_metrics.get("overall_risk_rating", "N/A"),
            "risk_flags": all_risk_flags,
            "confidence": avg_confidence,
            "key_metrics": all_metrics,
            "citations": list(set(all_citations)),
            "analysts_used": [r.analyst_type for r in analyst_results if not r.error],
        }

    def _format_analyst_inputs(self, results: List[AnalysisResult]) -> str:
        """Format all analyst results into a structured input for synthesis."""
        parts = []

        for result in results:
            if result.error:
                parts.append(
                    f"### {result.analyst_type.replace('_', ' ').title()}\n"
                    f"**Error**: {result.error}\n"
                )
                continue

            part = f"### {result.analyst_type.replace('_', ' ').title()}\n"
            part += f"**Confidence**: {result.confidence:.0%}\n\n"

            for section_name, section_content in result.sections.items():
                part += f"#### {section_name}\n{section_content}\n\n"

            if result.key_metrics:
                part += "#### Key Metrics\n"
                for k, v in result.key_metrics.items():
                    part += f"- {k}: {v}\n"
                part += "\n"

            if result.risk_flags:
                part += "#### Risk Flags\n"
                for flag in result.risk_flags:
                    part += f"- {flag}\n"
                part += "\n"

            parts.append(part)

        return "\n---\n\n".join(parts)

    def _fallback_report(
        self,
        results: List[AnalysisResult],
        question: str,
        company_name: str,
    ) -> str:
        """Generate a basic report when LLM synthesis fails."""
        lines = [
            f"# {company_name} SEC Filing Analysis Report",
            "",
            f"> **Analysis Query**: {question}",
            "",
            "---",
            "",
        ]

        for result in results:
            title = result.analyst_type.replace("_", " ").title()
            lines.append(f"## {title}")
            lines.append("")

            if result.error:
                lines.append(f"*Analysis unavailable: {result.error}*")
                lines.append("")
                continue

            for section_name, section_content in result.sections.items():
                lines.append(f"### {section_name}")
                lines.append(section_content)
                lines.append("")

        return "\n".join(lines)
