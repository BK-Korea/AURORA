"""Retriever node - searches vector store for relevant chunks."""
import re
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...vectorstore.chroma_store import ChromaStore, RetrievedChunk

logger = logging.getLogger(__name__)


class RetrieverNode:
    """
    Node that retrieves relevant document chunks for a query.

    Features:
    - Query optimization using LLM
    - Multi-query search for better recall
    - Semantic search using embeddings
    - Metadata filtering (form type, section, company, date)
    - Relevance scoring
    """

    QUERY_OPTIMIZER_PROMPT = """You are a search query optimizer for SEC financial documents.

CRITICAL RULES:
1. NEVER include company names (e.g., "Vertical", "Apple", "Tesla") in search queries
2. Focus ONLY on financial concepts and SEC terminology
3. Generate 2-3 short, focused English search queries

Financial terms to use:
- going concern, substantial doubt, ability to continue
- liquidity risk, cash burn, funding requirements
- net losses, operating cash flows, capital requirements
- risk factors, material uncertainty

Output ONLY the search queries, one per line. No company names. No explanations."""

    QUERY_CORRECTION_PROMPT = """당신은 사용자 질문을 교정하고 정규화하는 전문가입니다.

사용자가 타이핑한 질문을 분석하여:
1. 오타, 띄어쓰기 오류를 교정
2. 불완전한 단어를 완성
3. 다양한 표현을 표준 용어로 변환

## 규칙
- 원래 의도를 최대한 보존하면서 교정
- 띄어쓰기 오류 교정 (예: "팔 케 이" → "팔케이")
- 오타 교정 (예: "아처" → "아처" 또는 "Archer")
- 불완전한 단어 완성
- 한글/영문 혼용 정리

## 예시
- "archer의 팔 케 이에서" → "archer의 팔케이에서"
- "아처의 8-k" → "아처의 8-K"
- "조비의 십 케 이" → "조비의 십케이"

원본 질문: {question}

교정된 질문:"""

    UNIFIED_QUERY_INTERPRETATION_PROMPT = """당신은 SEC 문서 검색을 위한 질문 해석 전문가입니다.

사용자의 질문을 분석하여 다음 정보를 정확히 추출하고 정규화하세요.

**중요**: 모든 변형을 이해하세요. 매칭 테이블을 사용하지 말고, 문맥과 의미를 이해하여 변환하세요.

1. **회사명/티커**: 다양한 표현을 정확한 회사명이나 티커로 변환
   - "archer", "아쳐", "아처", "아 처" → "Archer" 또는 티커 "ACHR"
   - "joby", "조비", "조 비" → "Joby Aviation" 또는 티커 "JOBY"
   - 대소문자, 한글/영문 변형, 띄어쓰기 오류 모두 처리

2. **SEC Form Type**: 다양한 표현을 표준 Form Type으로 변환
   - "8-k", "팔케이", "팔 케 이", "팔케 이", "8K", "에잇케이" → "8-K"
   - "10-k", "십케이", "십 케 이", "연간보고서" → "10-K"
   - "10-q", "십큐", "십 큐", "분기보고서" → "10-Q"
   - 띄어쓰기, 오타, 한글/영문 혼용 모두 이해

3. **날짜 정보**: 최신, 최근, 특정 연도/월 등

## 출력 형식 (JSON)
```json
{{
  "company_name_or_ticker": "회사명 또는 티커 (없으면 null)",
  "form_type": "8-K, 10-K, 10-Q 등 (없으면 null)",
  "date_filter": "YYYY-MM-DD 또는 'latest' 또는 null",
  "interpretation_notes": "해석 과정의 주요 변환 사항"
}}
```

## 규칙
- 매칭 테이블 없이 문맥과 의미를 이해하여 변환
- 모든 띄어쓰기 변형, 오타, 불완전한 단어를 이해
- 한글/영문 변형, 대소문자 모두 고려
- 확실하지 않으면 null로 표시

질문: {question}

JSON 응답:"""

    COMPANY_EXTRACTION_PROMPT = """당신은 질문에서 회사명을 추출하는 전문가입니다.

사용자의 질문을 분석하여 언급된 회사명(또는 티커 심볼)을 추출하세요.

규칙:
1. 질문에 회사명이 명시적으로 언급되어 있으면 추출
2. 한국어 회사명, 영어 회사명, 티커 심볼 모두 가능
3. 회사명이 없으면 "NONE"이라고 답변
4. 회사명만 추출하고, 다른 설명은 하지 마세요

예시:
- "조비에 대해서 브리핑해줘" → "조비"
- "Apple의 재무상태를 분석해줘" → "Apple"
- "TSLA 주가에 대해 물어봐" → "TSLA"
- "최신 재무정보 알려줘" → "NONE"

질문: {question}

회사명:"""

    COMPANY_VERIFICATION_PROMPT = """당신은 질문과 인덱스된 회사 목록을 비교하여 정확한 회사를 확인하는 전문가입니다.

사용자의 다양한 표현을 이해하고 인덱스된 회사 목록과 정확히 매칭하세요.

## 규칙
1. 질문에서 언급된 회사명 또는 티커 심볼을 추출
2. 다양한 표현을 이해:
   - "archer", "아쳐", "아처" → "Archer Aviation" 또는 티커 "ACHR"
   - "joby", "조비" → "Joby Aviation" 또는 티커 "JOBY"
   - 대소문자, 한글/영문 변형 모두 처리
3. 인덱스된 회사 목록과 비교하여 가장 일치하는 회사를 선택
4. 회사명이 부분적으로 일치해도 괜찮습니다 (예: "Joby" → "Joby Aviation")
5. 티커 심볼도 매칭 가능합니다 (예: "JOBY", "AAPL", "TSLA", "ACHR")
6. 질문에 회사명이 없으면 "NONE"이라고 답변

## 출력 형식
정확한 회사명만 출력하세요. 인덱스된 회사 목록에 있는 정확한 이름을 사용하세요.

질문: {question}

인덱스된 회사 목록 (회사명 | 티커):
{indexed_companies}

질문에 해당하는 회사명 (인덱스된 목록에서 정확한 이름):"""

    TICKER_EXTRACTION_PROMPT = """당신은 질문에서 티커 심볼(주식 코드)을 추출하는 전문가입니다.

사용자의 질문을 분석하여 언급된 티커 심볼을 추출하세요.

## 규칙
1. 질문에 티커 심볼이 명시적으로 언급되어 있으면 추출 (예: "AAPL", "TSLA", "JOBY")
2. 티커 심볼은 보통 1-5자의 대문자 알파벳입니다
3. 일반적인 단어가 아닌 주식 코드만 추출하세요
4. 티커 심볼이 없으면 "NONE"이라고 답변
5. 티커 심볼만 추출하고, 다른 설명은 하지 마세요

예시:
- "AAPL의 재무상태를 분석해줘" → "AAPL"
- "TSLA 주가에 대해 물어봐" → "TSLA"
- "JOBY에 대해서 브리핑해줘" → "JOBY" (만약 JOBY가 티커라면)
- "Apple의 재무상태를 분석해줘" → "NONE" (회사명만 있고 티커 없음)
- "최신 재무정보 알려줘" → "NONE"

질문: {question}

티커 심볼:"""

    FORM_TYPE_EXTRACTION_PROMPT = """당신은 질문에서 SEC Form Type을 추출하고 정규화하는 전문가입니다.

**중요**: 매칭 테이블을 사용하지 말고, 문맥과 의미를 이해하여 모든 변형을 처리하세요.

사용자의 다양한 표현을 표준 SEC Form Type으로 변환하세요.

## 규칙
1. 모든 변형을 이해하고 표준 Form Type으로 변환 (매칭 테이블 없이)
2. 지원되는 Form Type: "10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A"
3. 모든 변형 처리 (띄어쓰기, 오타, 한글/영문 혼용):
   - "8-k", "팔케이", "팔 케 이", "팔케 이", "8K", "에잇케이" → "8-K"
   - "10-k", "십케이", "십 케 이", "연간보고서" → "10-K"
   - "10-q", "십큐", "십 큐", "분기보고서" → "10-Q"
4. 띄어쓰기 오류, 오타, 불완전한 단어 모두 이해
5. Form Type이 없으면 "NONE"이라고 답변
6. 표준 Form Type만 출력 (예: "8-K", "10-K")

예시:
- "8-k 중 기업조사보고서" → "8-K"
- "팔케이에서 찾아줘" → "8-K"
- "팔 케 이에서" → "8-K"
- "10-K의 재무상태를 분석해줘" → "10-K"
- "연간보고서" → "10-K"
- "10-Q 분기 보고서" → "10-Q"
- "재무정보 알려줘" → "NONE"

질문: {question}

SEC Form Type:"""

    def __init__(
        self,
        vector_store: ChromaStore,
        top_k: int = 10,
        min_score: float = 0.3,
        llm: Optional[BaseChatModel] = None,
        downloader: Optional[Any] = None,  # SECDownloader
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.llm = llm
        self.downloader = downloader
        self.progress_callback = progress_callback

    def correct_query(self, query: str) -> str:
        """
        Use LLM to correct user query (typos, spacing, incomplete words).
        
        This is a RAG-style query correction that fixes:
        - Typing errors
        - Spacing issues (e.g., "팔 케 이" → "팔케이")
        - Incomplete words
        - Mixed Korean/English
        
        Returns:
            Corrected query string
        """
        if not self.llm:
            return query
        
        try:
            messages = [
                SystemMessage(content="당신은 사용자 질문을 교정하는 전문가입니다. 오타, 띄어쓰기, 불완전한 단어를 교정하세요."),
                HumanMessage(content=self.QUERY_CORRECTION_PROMPT.format(question=query))
            ]
            response = self.llm.invoke(messages)
            corrected = response.content.strip()
            
            # Clean up response
            corrected = corrected.split('\n')[0].strip()
            
            if corrected and len(corrected) > 0:
                return corrected
        except Exception as e:
            logger.warning(f"Query correction failed: {e}")
        
        return query

    def interpret_query_unified(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to comprehensively interpret user query and extract/normalize all relevant information.
        
        Process:
        1. Single LLM call to both correct AND interpret query (optimized)
        
        This handles ALL variations without matching tables:
        - Company name variations (archer, 아쳐, 아처, 아 처 → Archer/ACHR)
        - Form type variations (8-k, 팔케이, 팔 케 이 → 8-K)
        - Date information
        
        Returns:
            Dict with normalized company_name_or_ticker, form_type, date_filter
        """
        if not self.llm:
            # Fallback to individual extractors
            return {
                "company_name_or_ticker": self.extract_company_with_llm(query),
                "form_type": self.extract_form_type_with_llm(query),
                "date_filter": self.extract_date_from_query(query),
                "interpretation_notes": None
            }
        
        # Single LLM call: correction + interpretation combined
        # This reduces API calls from 2 to 1
        try:
            messages = [
                SystemMessage(content="당신은 SEC 문서 검색을 위한 질문 교정 및 해석 전문가입니다. 먼저 질문의 오타와 띄어쓰기를 교정한 후, 회사명/티커, Form Type, 날짜를 추출하세요. 매칭 테이블 없이 문맥을 이해하여 변환하세요."),
                HumanMessage(content=f"""다음 질문을 교정하고 해석하세요:

원본 질문: {query}

1. 먼저 질문을 교정하세요 (오타, 띄어쓰기, 불완전한 단어)
2. 교정된 질문에서 회사명/티커, Form Type, 날짜를 추출하세요

{self.UNIFIED_QUERY_INTERPRETATION_PROMPT.format(question=query)}""")
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Log correction if visible in response
            if "교정" in content or "corrected" in content.lower():
                if self.progress_callback:
                    # Try to extract corrected query from response
                    import re
                    correction_match = re.search(r'교정[된]*\s*질문[:\s]*([^\n]+)', content)
                    if correction_match:
                        corrected = correction_match.group(1).strip()
                        self.progress_callback(f"✓ 질문 교정: '{query}' → '{corrected}'")
            
            # Try to parse JSON response
            import json
            import re
            
            # Extract JSON from response (handle markdown code blocks and nested JSON)
            # Try to find JSON object, handling both simple and nested structures
            json_str = None
            
            # Remove markdown code blocks if present
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object with balanced braces
                brace_count = 0
                start_idx = content.find('{')
                if start_idx != -1:
                    for i in range(start_idx, len(content)):
                        if content[i] == '{':
                            brace_count += 1
                        elif content[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = content[start_idx:i+1]
                                break
            
            if json_str:
                result = json.loads(json_str)
                
                # LLM이 이미 정규화된 Form Type을 반환했으므로, 단순히 검증만 수행
                # 매칭 테이블 사용하지 않음 - LLM이 모든 변형을 이해하여 변환
                if result.get("form_type"):
                    form_type = result["form_type"].strip()
                    
                    # LLM이 반환한 Form Type이 표준 형식인지 검증만 수행
                    valid_forms = ["10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A"]
                    if form_type not in valid_forms:
                        # LLM이 비표준 형식을 반환한 경우, 다시 LLM에 요청
                        # 하지만 일단은 그대로 사용 (LLM이 이미 이해했을 가능성)
                        logger.warning(f"LLM returned non-standard form type: {form_type}")
                        # 표준 형식으로 변환 시도 (최소한의 정규화)
                        form_type_upper = form_type.upper().replace(" ", "-")
                        if form_type_upper in valid_forms:
                            form_type = form_type_upper
                        else:
                            form_type = None
                    
                    result["form_type"] = form_type
                
                # Normalize company name/ticker (handle null, "NONE", etc.)
                company_or_ticker = result.get("company_name_or_ticker")
                if company_or_ticker:
                    company_or_ticker = company_or_ticker.strip()
                    if company_or_ticker.upper() in ["NONE", "NULL", "없음", "없습니다"]:
                        company_or_ticker = None
                    else:
                        # Clean up
                        company_or_ticker = company_or_ticker.split('\n')[0].strip()
                        company_or_ticker = company_or_ticker.split('.')[0].strip()
                
                return {
                    "company_name_or_ticker": company_or_ticker if company_or_ticker else None,
                    "form_type": result.get("form_type"),
                    "date_filter": result.get("date_filter"),
                    "interpretation_notes": result.get("interpretation_notes")
                }
        except Exception as e:
            logger.warning(f"Unified query interpretation failed: {e}, falling back to individual extractors")
        
        # Fallback to individual extractors
        return {
            "company_name_or_ticker": self.extract_company_with_llm(query),
            "form_type": self.extract_form_type_with_llm(query),
            "date_filter": self.extract_date_from_query(query),
            "interpretation_notes": None
        }

    def extract_company_with_llm(self, query: str) -> Optional[str]:
        """
        Use LLM to extract company name from query.
        
        Returns:
            Company name if found, None otherwise
        """
        if not self.llm:
            # Fallback to regex if no LLM
            return self.extract_company_from_query(query)
        
        try:
            messages = [
                SystemMessage(content="당신은 질문에서 회사명을 정확히 추출하는 전문가입니다."),
                HumanMessage(content=self.COMPANY_EXTRACTION_PROMPT.format(question=query))
            ]
            response = self.llm.invoke(messages)
            company = response.content.strip()
            
            # Check if LLM said "NONE" or similar
            if company.upper() in ["NONE", "없음", "없습니다", "회사명 없음"]:
                return None
            
            # Clean up response (remove explanations)
            company = company.split('\n')[0].strip()
            company = company.split('.')[0].strip()
            company = company.split('(')[0].strip()
            
            if len(company) > 0 and company.upper() != "NONE":
                return company
        except Exception as e:
            # Fallback to regex on error
            pass
        
        return self.extract_company_from_query(query)

    def extract_ticker_with_llm(self, query: str) -> Optional[str]:
        """
        Use LLM to extract ticker symbol from query.
        
        Returns:
            Ticker symbol if found, None otherwise
        """
        if not self.llm:
            return None
        
        try:
            messages = [
                SystemMessage(content="당신은 질문에서 티커 심볼을 정확히 추출하는 전문가입니다."),
                HumanMessage(content=self.TICKER_EXTRACTION_PROMPT.format(question=query))
            ]
            response = self.llm.invoke(messages)
            ticker = response.content.strip()
            
            # Check if LLM said "NONE" or similar (case-insensitive)
            ticker_upper = ticker.upper()
            if ticker_upper in ["NONE", "없음", "없습니다", "티커 없음", "N/A"]:
                return None
            
            # Clean up response (remove explanations)
            ticker = ticker.split('\n')[0].strip()
            ticker = ticker.split('.')[0].strip()
            ticker = ticker.split('(')[0].strip()
            ticker = ticker.upper()  # Normalize to uppercase
            
            # Validate ticker format (1-5 alphanumeric characters)
            if len(ticker) > 0 and len(ticker) <= 5 and ticker.isalnum():
                return ticker
        except Exception as e:
            logger.warning(f"Ticker extraction failed: {e}")
        
        return None

    def verify_company_with_indexed_list(
        self, 
        query: str, 
        indexed_companies: List[str],
        indexed_tickers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Use LLM to verify and match company from question against indexed companies.
        
        This ensures we use the exact company name as stored in the index.
        Also checks for ticker symbols if provided.
        
        Args:
            query: User's question
            indexed_companies: List of company names in the index
            indexed_tickers: Optional dict mapping company_name -> ticker
            
        Returns:
            Exact company name from indexed_companies if match found, None otherwise
        """
        if not self.llm or not indexed_companies:
            return None
        
        try:
            # First, try to extract ticker from query
            extracted_ticker = self.extract_ticker_with_llm(query)
            
            # If ticker found and we have ticker mapping, use it
            if extracted_ticker and indexed_tickers:
                for company_name, ticker in indexed_tickers.items():
                    if ticker and ticker.upper() == extracted_ticker.upper():
                        if self.progress_callback:
                            self.progress_callback(f"✓ 티커로 회사 매칭: {extracted_ticker} → {company_name}")
                        return company_name
            
            # Build companies list with tickers if available
            if indexed_tickers:
                companies_list = "\n".join([
                    f"- {c} | {indexed_tickers.get(c, 'N/A')}" 
                    for c in indexed_companies
                ])
            else:
                companies_list = "\n".join([f"- {c}" for c in indexed_companies])
            
            messages = [
                SystemMessage(content="당신은 질문과 인덱스된 회사 목록을 비교하여 정확한 회사를 찾는 전문가입니다."),
                HumanMessage(content=self.COMPANY_VERIFICATION_PROMPT.format(
                    question=query,
                    indexed_companies=companies_list
                ))
            ]
            response = self.llm.invoke(messages)
            verified_company = response.content.strip()
            
            # Check if LLM said "NONE" or similar
            if verified_company.upper() in ["NONE", "없음", "없습니다", "회사명 없음"]:
                return None
            
            # Clean up response
            verified_company = verified_company.split('\n')[0].strip()
            verified_company = verified_company.split('.')[0].strip()
            verified_company = verified_company.split('(')[0].strip()
            
            # Verify the company is actually in the indexed list (fuzzy match)
            from thefuzz import fuzz
            best_match = None
            best_score = 0
            
            for indexed_company in indexed_companies:
                score = fuzz.ratio(
                    verified_company.lower(),
                    indexed_company.lower()
                )
                if score > best_score:
                    best_score = score
                    best_match = indexed_company
            
            # If match is good (>= 70%), return the exact indexed company name
            if best_score >= 70 and best_match:
                return best_match
            
        except Exception as e:
            logger.warning(f"Company verification failed: {e}")
        
        return None

    @staticmethod
    def extract_company_from_query(query: str) -> Optional[str]:
        """Extract company name from query if mentioned."""
        # Common patterns: "조비에 대해서", "Apple의", "Tesla 회사", etc.
        # This is a simple heuristic - can be improved with LLM
        query_lower = query.lower()
        
        # Check for common Korean patterns
        korean_patterns = [
            r'([가-힣]+(?:\([A-Z]+\))?)\s*(?:에\s*대해서|의|회사|기업)',
            r'([가-힣]+)\s*(?:브리핑|분석|정보)',
        ]
        
        for pattern in korean_patterns:
            match = re.search(pattern, query)
            if match:
                company = match.group(1).strip()
                # Remove common suffixes
                company = re.sub(r'\([^)]+\)', '', company).strip()
                if len(company) > 1:
                    return company
        
        return None

    def extract_form_type_with_llm(self, query: str) -> Optional[str]:
        """
        Use LLM to extract SEC form type from query.
        
        Returns:
            Form type if found (e.g., "8-K", "10-K"), None otherwise
        """
        if not self.llm:
            return None
        
        try:
            messages = [
                SystemMessage(content="당신은 질문에서 SEC Form Type을 정확히 추출하는 전문가입니다."),
                HumanMessage(content=self.FORM_TYPE_EXTRACTION_PROMPT.format(question=query))
            ]
            response = self.llm.invoke(messages)
            form_type = response.content.strip()
            
            # Check if LLM said "NONE" or similar
            form_type_upper = form_type.upper()
            if form_type_upper in ["NONE", "없음", "없습니다", "N/A"]:
                return None
            
            # Clean up response
            form_type = form_type.split('\n')[0].strip()
            form_type = form_type.split('.')[0].strip()
            form_type = form_type.split('(')[0].strip()
            
            # LLM이 이미 정규화했을 것으로 가정하고, 최소한의 정규화만 수행
            # 매칭 테이블 사용하지 않음 - LLM이 모든 변형을 이해하여 변환
            form_type = form_type.strip()
            
            # 표준 형식으로 변환 (띄어쓰기 제거, 대문자 변환)
            # LLM이 "8-K", "10-K" 등 표준 형식으로 반환했는지 확인
            form_type_normalized = form_type.upper().replace(" ", "-")
            
            # Validate form type (LLM이 이미 올바른 형식으로 반환했는지 확인)
            valid_forms = ["10-K", "10-Q", "8-K", "20-F", "6-K", "DEF 14A"]
            if form_type_normalized in valid_forms:
                return form_type_normalized
            
            # LLM이 비표준 형식을 반환한 경우, None 반환
            # (통합 해석에서 처리할 것)
            return None
            
        except Exception as e:
            logger.warning(f"Form type extraction failed: {e}")
        
        return None

    @staticmethod
    def extract_date_from_query(query: str, current_year: int = None) -> Optional[str]:
        """Extract date information from query and return minimum date filter (YYYY-MM-DD)."""
        if current_year is None:
            current_year = datetime.now().year
        
        # Patterns: "26년 1월", "2026년 1월", "최신", "최근"
        query_lower = query.lower()
        
        # Check for "최신" or "최근" - use current year
        if "최신" in query_lower or "최근" in query_lower:
            return f"{current_year}-01-01"
        
        # Pattern: "26년 1월" or "2026년 1월"
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월',  # 2026년 1월
            r'(\d{2})년\s*(\d{1,2})월',  # 26년 1월
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                year_str, month_str = match.groups()
                if len(year_str) == 2:
                    # Convert 26 -> 2026
                    year = 2000 + int(year_str)
                else:
                    year = int(year_str)
                month = int(month_str)
                return f"{year}-{month:02d}-01"
        
        return None


    def _optimize_query(self, query: str) -> List[str]:
        """Use LLM to generate optimized search queries."""
        if not self.llm:
            return [query]
        
        try:
            messages = [
                SystemMessage(content=self.QUERY_OPTIMIZER_PROMPT),
                HumanMessage(content=f"User question: {query}")
            ]
            response = self.llm.invoke(messages)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            # Always include original query as fallback
            if query not in queries:
                queries.append(query)
            return queries[:3]  # Max 3 queries
        except Exception:
            return [query]

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for the current query.

        Input state:
            - current_query: User's question
            - company_info: Optional CompanyInfo for filtering

        Output state:
            - retrieved_chunks: List of RetrievedChunk
            - error: Error message if retrieval failed
        """
        query = state.get("current_query", "")

        if not query:
            return {
                "retrieved_chunks": [],
                "error": "질문을 입력해주세요."
            }

        # Get filters from state (set by unified interpretation in ask())
        # This uses the LangChain-based unified interpretation that handles all variations
        indexed_company_name = state.get("indexed_company_name")
        company_info = state.get("company_info")
        filter_form_type = state.get("filter_form_type")  # From unified interpretation
        filter_date = state.get("filter_date")  # From unified interpretation
        
        # Fallback: if not in state, extract from query (backward compatibility)
        if not filter_form_type:
            filter_form_type = self.extract_form_type_with_llm(query)
        if not filter_date:
            filter_date = self.extract_date_from_query(query)
        
        # Use indexed_company_name from state (exact match for filtering)
        # Otherwise use company_info.name
        filter_company_name = None
        
        # Priority: indexed_company_name > company_info.name
        if indexed_company_name:
            # Use exact indexed company name for ChromaDB filtering
            filter_company_name = indexed_company_name
        elif company_info:
            filter_company_name = company_info.name

        # Log filters if found
        if filter_form_type and self.progress_callback:
            self.progress_callback(f"✓ Form Type 필터링: {filter_form_type}만 검색")
        if filter_date and self.progress_callback:
            self.progress_callback(f"✓ 날짜 필터링: {filter_date} 이후")

        # Optimize query using LLM
        search_queries = self._optimize_query(query)
        
        # Perform multi-query search with filters
        all_chunks = {}
        filter_form_types = [filter_form_type] if filter_form_type else None
        
        for sq in search_queries:
            chunks = self.vector_store.search(
                query=sq,
                top_k=self.top_k,
                min_score=self.min_score,
                filter_company_name=filter_company_name,
                filter_min_date=filter_date,
                filter_form_types=filter_form_types
            )
            for chunk in chunks:
                # Use content hash as key to deduplicate
                key = hash(chunk.content)
                if key not in all_chunks or chunk.score > all_chunks[key].score:
                    all_chunks[key] = chunk
        
        # Sort by score and take top results
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
        result_chunks = sorted_chunks[:self.top_k]

        if not result_chunks:
            error_msg = "관련 정보를 찾을 수 없습니다."
            if filter_company_name:
                error_msg += f" '{filter_company_name}' 회사의 문서가 인덱스에 없을 수 있습니다."
            if filter_date:
                error_msg += f" {filter_date} 이후의 최신 문서가 없을 수 있습니다."
            return {
                "retrieved_chunks": [],
                "error": error_msg + " 다른 질문을 시도해주세요."
            }

        return {
            "retrieved_chunks": result_chunks,
            "error": None
        }


class ContextBuilder:
    """Helper class to build context from retrieved chunks."""

    @staticmethod
    def build_context(
        chunks: List[RetrievedChunk],
        max_tokens: int = 8000,
        include_scores: bool = False
    ) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of retrieved chunks
            max_tokens: Approximate max tokens (chars / 4)
            include_scores: Include relevance scores

        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4

        for chunk in chunks:
            chunk_text = chunk.to_context_string()
            if include_scores:
                chunk_text = f"[Relevance: {chunk.score:.2f}]\n{chunk_text}"

            if total_chars + len(chunk_text) > max_chars:
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)
