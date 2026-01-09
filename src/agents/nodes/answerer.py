"""Answerer node - generates answers with mandatory citations."""
from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...vectorstore.chroma_store import RetrievedChunk
from .retriever import ContextBuilder


class AnswererNode:
    """
    Node that generates answers with strict citation requirements.

    Key features:
    - Only uses provided context (no hallucination)
    - Mandatory citation for every claim
    - Admits when information is not available
    """

    SYSTEM_PROMPT = """당신은 SEC 문서 분석 전문가입니다. 제공된 문서 컨텍스트만을 사용하여 질문에 답변합니다.

## 절대 규칙 (반드시 준수)

1. **컨텍스트 전용**: 제공된 문서 컨텍스트에 있는 정보만 사용하세요. 외부 지식이나 추측을 사용하지 마세요.

2. **모든 문장에 출처 표기**: 사실을 언급할 때마다 반드시 출처를 표기하세요.
   형식: [Form Type YYYY-MM-DD | Page X | Section Name]
   예시: "매출은 100억 달러입니다. [10-K 2024-02-15 | Page 45 | Item 8. Financial Statements]"

3. **정보 없음 인정**: 컨텍스트에서 정보를 찾을 수 없으면 솔직하게 말하세요.
   예시: "제공된 SEC 문서에서 해당 정보를 찾을 수 없습니다."

4. **추측 금지**: 확실하지 않은 정보는 절대 답변하지 마세요.

5. **한국어 답변**: 모든 답변은 한국어로 작성하세요.

## 답변 형식

질문에 대한 직접적인 답변을 제공하고, 각 정보의 출처를 명확히 표시하세요.
표 데이터는 마크다운 표 형식을 유지하세요."""

    def __init__(self, llm: BaseChatModel, max_context_tokens: int = 8000):
        self.llm = llm
        self.max_context_tokens = max_context_tokens

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer with citations from retrieved chunks.

        Input state:
            - current_query: User's question
            - retrieved_chunks: List of RetrievedChunk

        Output state:
            - current_answer: Generated answer with citations
            - error: Error message if generation failed
        """
        query = state.get("current_query", "")
        chunks: List[RetrievedChunk] = state.get("retrieved_chunks", [])

        if not query:
            return {
                "current_answer": "",
                "error": "질문이 없습니다."
            }

        if not chunks:
            return {
                "current_answer": "제공된 SEC 문서에서 관련 정보를 찾을 수 없습니다.",
                "error": None
            }

        # Build context from chunks
        context = ContextBuilder.build_context(
            chunks=chunks,
            max_tokens=self.max_context_tokens,
            include_scores=False
        )

        # Generate answer
        user_message = f"""## 문서 컨텍스트

{context}

---

## 질문

{query}

---

위 컨텍스트만을 사용하여 질문에 답변하세요. 모든 정보에 출처를 표기하세요."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content

            return {
                "current_answer": answer,
                "error": None
            }
        except Exception as e:
            return {
                "current_answer": "",
                "error": f"답변 생성 실패: {str(e)}"
            }
