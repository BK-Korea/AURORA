"""Company resolver node - matches company names to SEC tickers."""
from typing import Dict, Any, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ...document.sec_downloader import SECDownloader, CompanyInfo


class CompanyResolverNode:
    """
    Node that resolves company names to SEC CIK/ticker.

    Uses fuzzy matching + LLM disambiguation for ambiguous cases.
    """

    SYSTEM_PROMPT = """당신은 회사명을 SEC 티커로 매칭하는 전문가입니다.
사용자가 입력한 회사명과 가장 일치하는 회사를 선택하세요.
숫자만 응답하세요."""

    def __init__(self, downloader: SECDownloader, llm: BaseChatModel):
        self.downloader = downloader
        self.llm = llm

    def _llm_disambiguate(self, query: str, candidates: List[CompanyInfo]) -> int:
        """Use LLM to pick the best match from candidates."""
        options = "\n".join([
            f"{i+1}. {c.name} ({c.ticker})"
            for i, c in enumerate(candidates)
        ])

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"검색어: {query}\n\n옵션:\n{options}\n\n가장 적합한 번호:")
        ]

        response = self.llm.invoke(messages)
        try:
            choice = int(response.content.strip()) - 1
            if 0 <= choice < len(candidates):
                return choice
        except:
            pass
        return 0

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve company query to CompanyInfo.

        Input state:
            - company_query: User's company search query

        Output state:
            - company_info: Resolved CompanyInfo or None
            - error: Error message if resolution failed
        """
        query = state.get("company_query", "")

        if not query:
            return {
                "company_info": None,
                "error": "회사명을 입력해주세요."
            }

        # Search for candidates
        candidates = self.downloader.search_company(query, limit=5)

        if not candidates:
            return {
                "company_info": None,
                "error": f"'{query}'에 해당하는 회사를 찾을 수 없습니다."
            }

        # Single match
        if len(candidates) == 1:
            return {
                "company_info": candidates[0],
                "error": None
            }

        # Exact ticker match
        query_upper = query.strip().upper()
        for c in candidates:
            if c.ticker and c.ticker.upper() == query_upper:
                return {
                    "company_info": c,
                    "error": None
                }

        # LLM disambiguation
        best_idx = self._llm_disambiguate(query, candidates)
        return {
            "company_info": candidates[best_idx],
            "error": None
        }
