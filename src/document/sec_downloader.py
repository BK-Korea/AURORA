"""SEC EDGAR document downloader with company name resolution."""
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import httpx
from thefuzz import fuzz, process
from rich.console import Console

console = Console()


@dataclass
class CompanyInfo:
    """Company information from SEC."""
    cik: str
    name: str
    ticker: Optional[str]


@dataclass
class SECFiling:
    """SEC filing metadata."""
    accession_number: str
    form_type: str
    filing_date: str
    file_path: Path
    description: str = ""


class SECDownloader:
    """Download SEC filings with fuzzy company name matching."""

    SEC_BASE_URL = "https://www.sec.gov"
    EDGAR_COMPANY_SEARCH = "https://www.sec.gov/cgi-bin/browse-edgar"
    EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    # US companies: 10-K, 10-Q, 8-K
    # Foreign companies: 20-F (annual), 6-K (current)
    SUPPORTED_FORMS = ["10-K", "10-Q", "8-K", "DEF 14A", "20-F", "6-K"]

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._company_tickers: Optional[Dict] = None
        self._headers = {
            "User-Agent": "AURORA Research Tool (academic research) contact@example.com",
            "Accept-Encoding": "gzip, deflate",
        }

    def _load_company_tickers(self) -> Dict:
        """Load company tickers from SEC or cache."""
        cache_file = self.data_dir / "company_tickers.json"

        # Use cache if less than 7 days old
        if cache_file.exists():
            cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if cache_age < 7 * 24 * 3600:
                with open(cache_file, "r") as f:
                    return json.load(f)

        # Download fresh data
        try:
            with httpx.Client(headers=self._headers, timeout=30) as client:
                response = client.get(self.COMPANY_TICKERS_URL)
                response.raise_for_status()
                data = response.json()

                # Cache it
                with open(cache_file, "w") as f:
                    json.dump(data, f)

                return data
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch company tickers: {e}[/yellow]")
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)
            return {}

    @property
    def company_tickers(self) -> Dict:
        """Get company tickers data (lazy loaded)."""
        if self._company_tickers is None:
            self._company_tickers = self._load_company_tickers()
        return self._company_tickers

    def search_company(self, query: str, limit: int = 5) -> List[CompanyInfo]:
        """
        Search for companies by name or ticker using fuzzy matching.

        Args:
            query: Company name or ticker to search
            limit: Maximum number of results to return

        Returns:
            List of matching CompanyInfo objects
        """
        query = query.strip().upper()
        results = []

        # Build searchable list
        companies = []
        for idx, item in self.company_tickers.items():
            ticker = item.get("ticker", "")
            name = item.get("title", "")
            cik = str(item.get("cik_str", "")).zfill(10)
            companies.append({
                "ticker": ticker,
                "name": name,
                "cik": cik,
                "search_text": f"{ticker} {name}".upper()
            })

        # Exact ticker match first
        for c in companies:
            if c["ticker"].upper() == query:
                results.append(CompanyInfo(
                    cik=c["cik"],
                    name=c["name"],
                    ticker=c["ticker"]
                ))
                if len(results) >= limit:
                    return results

        # Fuzzy match on name and ticker
        search_texts = {c["search_text"]: i for i, c in enumerate(companies)}
        matches = process.extract(
            query,
            list(search_texts.keys()),
            scorer=fuzz.token_set_ratio,
            limit=limit * 2
        )

        for match in matches:
            match_text, score = match[0], match[1]
            if score < 50:
                continue
            idx = search_texts.get(match_text)
            if idx is None:
                continue
            c = companies[idx]
            info = CompanyInfo(
                cik=c["cik"],
                name=c["name"],
                ticker=c["ticker"]
            )
            if info not in results:
                results.append(info)
            if len(results) >= limit:
                break

        return results

    def resolve_company(
        self,
        query: str,
        llm_callback: Optional[callable] = None
    ) -> Optional[CompanyInfo]:
        """
        Resolve company name to CIK using fuzzy matching and optional LLM.

        Args:
            query: Company name or ticker
            llm_callback: Optional LLM function for disambiguation

        Returns:
            CompanyInfo if found, None otherwise
        """
        matches = self.search_company(query, limit=5)

        if not matches:
            return None

        # Exact match
        if len(matches) == 1:
            return matches[0]

        # Check if top match is significantly better
        if matches[0].ticker and matches[0].ticker.upper() == query.upper():
            return matches[0]

        # Use LLM for disambiguation if available
        if llm_callback and len(matches) > 1:
            options = "\n".join([
                f"{i+1}. {m.name} ({m.ticker})" for i, m in enumerate(matches)
            ])
            prompt = f"""사용자가 "{query}" 회사를 검색했습니다.
다음 중 가장 적합한 회사를 선택하세요:
{options}

숫자만 답하세요 (예: 1)"""

            try:
                response = llm_callback(prompt)
                choice = int(re.search(r'\d+', response).group()) - 1
                if 0 <= choice < len(matches):
                    return matches[choice]
            except:
                pass

        # Return best match
        return matches[0]

    def download_filings(
        self,
        cik: str,
        form_types: List[str] = None,
        years: int = 3,
        progress_callback: Optional[callable] = None
    ) -> List[SECFiling]:
        """
        Download SEC filings for a company.

        Args:
            cik: Company CIK number
            form_types: List of form types to download (default: all supported)
            years: Number of years of filings to download
            progress_callback: Optional callback for progress updates

        Returns:
            List of downloaded SECFiling objects
        """
        if form_types is None:
            form_types = self.SUPPORTED_FORMS

        cik = cik.lstrip("0")
        filings = []

        # Get company filings index
        submissions_url = f"{self.SEC_BASE_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=100&output=atom"

        try:
            with httpx.Client(headers=self._headers, timeout=60) as client:
                # Get submissions
                submissions_api_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
                response = client.get(submissions_api_url)
                response.raise_for_status()
                data = response.json()

                company_name = data.get("name", "Unknown")
                recent_filings = data.get("filings", {}).get("recent", {})

                if not recent_filings:
                    return filings

                # Create company directory
                company_dir = self.raw_dir / cik
                company_dir.mkdir(exist_ok=True)

                # Process filings
                form_list = recent_filings.get("form", [])
                date_list = recent_filings.get("filingDate", [])
                accession_list = recent_filings.get("accessionNumber", [])
                primary_doc_list = recent_filings.get("primaryDocument", [])

                cutoff_year = datetime.now().year - years

                for i, (form, date, accession, primary_doc) in enumerate(
                    zip(form_list, date_list, accession_list, primary_doc_list)
                ):
                    # Check form type
                    if form not in form_types:
                        continue

                    # Check date
                    filing_year = int(date.split("-")[0])
                    if filing_year < cutoff_year:
                        continue

                    # Download filing
                    accession_clean = accession.replace("-", "")
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_doc}"

                    try:
                        doc_response = client.get(doc_url)
                        doc_response.raise_for_status()

                        # Save file
                        file_name = f"{form.replace(' ', '_')}_{date}_{accession}.html"
                        file_path = company_dir / file_name

                        with open(file_path, "wb") as f:
                            f.write(doc_response.content)

                        filing = SECFiling(
                            accession_number=accession,
                            form_type=form,
                            filing_date=date,
                            file_path=file_path,
                            description=f"{company_name} {form} filed {date}"
                        )
                        filings.append(filing)

                        if progress_callback:
                            progress_callback(f"Downloaded: {form} ({date})")

                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to download {form} {date}: {e}[/yellow]")

        except Exception as e:
            console.print(f"[red]Error fetching filings: {e}[/red]")

        return filings

    def get_downloaded_filings(self, cik: str) -> List[SECFiling]:
        """Get list of already downloaded filings for a company."""
        company_dir = self.raw_dir / cik
        if not company_dir.exists():
            return []

        filings = []
        for file_path in company_dir.glob("*.html"):
            # Parse filename: FORM_DATE_ACCESSION.html
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                form_type = parts[0].replace("_", " ")
                filing_date = parts[1]
                accession = "_".join(parts[2:])

                filings.append(SECFiling(
                    accession_number=accession,
                    form_type=form_type,
                    filing_date=filing_date,
                    file_path=file_path
                ))

        return sorted(filings, key=lambda x: x.filing_date, reverse=True)
