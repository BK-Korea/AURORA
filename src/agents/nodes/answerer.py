"""Answerer node - generates answers with mandatory citations."""
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...vectorstore.chroma_store import RetrievedChunk
from .retriever import ContextBuilder
from .quality_refiner import IterativeAnswerer

logger = logging.getLogger(__name__)


class AnswererNode:
    """
    Node that generates CEO-quality answers through iterative refinement.

    Key features:
    - Iterative quality improvement (generate → evaluate → refine)
    - Target score: 8/10 for CEO-level quality
    - Mandatory citations with English quotes
    - Rich financial analysis
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_context_tokens: int = 16000,
        target_score: int = 8,
        max_iterations: int = 2,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.llm = llm
        self.max_context_tokens = max_context_tokens
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.progress_callback = progress_callback
        
        # Initialize iterative answerer
        self.iterative_answerer = IterativeAnswerer(
            llm=llm,
            target_score=target_score,
            max_iterations=max_iterations
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate high-quality answer with citations from retrieved chunks.

        Input state:
            - current_query: User's question
            - retrieved_chunks: List of RetrievedChunk

        Output state:
            - current_answer: Generated answer with citations
            - answer_score: Quality score (1-10)
            - error: Error message if generation failed
        """
        query = state.get("current_query", "")
        chunks: List[RetrievedChunk] = state.get("retrieved_chunks", [])

        if not query:
            return {
                "current_answer": "",
                "answer_score": 0,
                "error": "질문이 없습니다."
            }

        if not chunks:
            return {
                "current_answer": "제공된 SEC 문서에서 관련 정보를 찾을 수 없습니다.",
                "answer_score": 0,
                "error": None
            }

        # Build context from chunks
        context = ContextBuilder.build_context(
            chunks=chunks,
            max_tokens=self.max_context_tokens,
            include_scores=False
        )

        try:
            # Verify all chunks are from the same company (if company_info is in state)
            company_info = state.get("company_info")
            if company_info:
                chunk_companies = set()
                for chunk in chunks:
                    chunk_company = chunk.metadata.get("company_name")
                    if chunk_company:
                        chunk_companies.add(chunk_company)
                
                # Check if all chunks match the expected company
                if chunk_companies and company_info.name not in chunk_companies:
                    # Fuzzy match check
                    from thefuzz import fuzz
                    best_match = None
                    best_score = 0
                    for chunk_company in chunk_companies:
                        score = fuzz.partial_ratio(
                            company_info.name.lower(),
                            chunk_company.lower()
                        )
                        if score > best_score:
                            best_score = score
                            best_match = chunk_company
                    
                    if best_score < 70:  # Low match threshold
                        return {
                            "current_answer": "",
                            "answer_score": 0,
                            "error": (
                                f"질문한 회사 '{company_info.name}'와 인덱스된 문서의 회사가 일치하지 않습니다. "
                                f"인덱스된 문서: {', '.join(list(chunk_companies)[:3])}"
                            )
                        }

            # Use iterative answerer for high-quality output
            answer, score, iterations = self.iterative_answerer.generate(
                question=query,
                context=context,
                progress_callback=self.progress_callback
            )

            # Final verification: Check if answer mentions the correct company
            if company_info:
                answer_lower = answer.lower()
                company_name_lower = company_info.name.lower()
                ticker_lower = (company_info.ticker or "").lower()
                
                # Check if answer mentions the company
                mentions_company = (
                    company_name_lower in answer_lower or
                    ticker_lower in answer_lower or
                    any(word in answer_lower for word in company_name_lower.split() if len(word) > 3)
                )
                
                # If answer doesn't mention the company at all, add a warning
                if not mentions_company and len(answer) > 200:
                    # This might indicate the answer is about a different company
                    logger.warning(
                        f"Answer doesn't mention company {company_info.name} but company was specified in query"
                    )

            return {
                "current_answer": answer,
                "answer_score": score,
                "iterations_used": iterations,
                "error": None
            }
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"Answer generation failed: {error_detail}")
            print(f"\n[DEBUG] Error in answerer: {str(e)}", flush=True)
            return {
                "current_answer": "",
                "answer_score": 0,
                "error": f"답변 생성 실패: {str(e)}"
            }
