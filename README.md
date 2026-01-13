# AURORA - SEC Company Research Tool

```
     █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗
    ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗
    ███████║██║   ██║██████╔╝██║   ██║██████╔╝███████║
    ██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══██║
    ██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

        ◈ Autonomous Research & Analysis System ◈
```

AI-powered SEC document analysis with **CEO-level quality reports** and strict citation verification.

## ✨ Features

### Core Capabilities
- **Company Resolution**: Fuzzy matching for company names/tickers (supports 13,000+ SEC-registered companies)
- **SEC EDGAR Integration**: Automatic download of 10-K, 10-Q, 8-K filings
- **Foreign Company Support**: 20-F, 6-K filings for international companies
- **Table Preservation**: Markdown conversion for financial tables
- **Persistent Index**: Documents are indexed once and reused across sessions

### AI-Powered Analysis
- **CEO-Level Reports**: Structured analysis with Executive Summary, Detailed Analysis, and Risk Factors
- **Iterative Quality Refinement**: Self-evaluating system that improves answers until 8+/10 quality score
- **LLM Query Optimization**: Automatically generates multiple semantic search queries for better retrieval
- **Citation Verification**: Every claim is traced to source documents with page numbers
- **Direct English Quotes**: Key findings include original English text from SEC filings
- **No Hallucination**: Strict RAG with source-only responses
- **Intelligent Query Interpretation**: Handles various linguistic forms (e.g., "archer"/"아쳐", "8-k"/"팔케이") using LangChain-based normalization
- **Smart Filtering**: Filter by company, form type (10-K, 10-Q, 8-K, etc.), and date range
- **Performance Optimized**: Reduced LLM API calls through unified query interpretation and fuzzy matching

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/BK-Korea/AURORA.git
cd AURORA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Required: LLM (Zhipu AI GLM-4.7)
GLM_API_KEY=your_glm_api_key_here

# Recommended: OpenAI Embeddings (higher quality, more stable)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Use GLM embeddings instead (if no OpenAI key)
# System will automatically fall back to GLM embeddings
```

### Embedding Options

| Provider | Model | Pros | Cons |
|----------|-------|------|------|
| **OpenAI** (Recommended) | text-embedding-3-small | High quality, stable, fast | Requires API key |
| GLM (Zhipu AI) | embedding-2 | Single API key | May have availability issues |

## 📖 Usage

### Research a Company

```bash
# Interactive mode
aurora research

# Direct company lookup
aurora research "Apple"
aurora research AAPL --years 5

# Foreign companies (automatically uses 20-F, 6-K)
aurora research "Vertical Aerospace"
```

### Ask Questions

```bash
# Quick question on indexed documents
aurora ask "What are the main risk factors?"

# Detailed financial analysis
aurora ask "주요 비용의 년간 변화추이를 분석해줘"

# Filter by form type (supports various formats)
aurora ask "archer의 8-k에서 중요 내용 정리해줘"
aurora ask "아쳐의 팔케이 중 기업조사보고서에 포함되어야할 중요내용들을 뽑아서 정리해줘"

# Filter by date
aurora ask "최신 정보들에 대해서 브리핑 해줘. 지금은 26년 1월이야"

# Combined filters
aurora ask "Apple의 10-K 최신 재무상태를 분석해줘"
```

**Query Interpretation Features:**
- **Multi-language Support**: Handles Korean, English, and mixed inputs
- **Form Type Normalization**: Automatically converts "8-k", "팔케이", "팔 케 이" → "8-K"
- **Company Name Variations**: Matches "archer", "아쳐", "아 처" → "Archer"
- **Ticker Symbol Support**: Recognizes both company names and ticker symbols (e.g., "AAPL", "ACHR")
- **Date Filtering**: Understands relative dates ("최신", "26년 1월") and converts to date filters

### Other Commands

```bash
# Check index status
aurora status

# Clear all indexed documents
aurora clear
```

## 📊 Example Output

```
╭─────────────────────────── Answer (Quality: 9/10) ───────────────────────────╮
│                                                                              │
│  보고서 수신: CEO                                                            │
│  주제: Vertical Aero 주요 비용의 연간 변화 추이 및 재무적 함의 분석          │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                       1. Executive Summary (핵심 요약)                       │
│                                                                              │
│  Vertical Aero의 연구개발(R&D) 비용은 2020년부터 2022년까지 3년간 약 172%    │
│  급증하였으며, 판매관리비(SG&A)는 동일 기간 약 308% 증가...                  │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                 2. 상세 분석                                 │
│                                                                              │
│   • 2020년: $108,741 (천 달러) [10-K 2023-03-01 | Page 50 | Item 8]          │
│   • 2021년: $197,568 (천 달러) [10-K 2023-03-01 | Page 50 | Item 8]          │
│   • 2022년: $296,281 (천 달러) [10-K 2023-03-01 | Page 50 | Item 8]          │
│                                                                              │
│  ▌ "We expect our research and development expenses to increase as we        │
│  ▌ increase staffing to support aircraft engineering and software            │
│  ▌ development..." [10-K 2024-02-27 | Page 41 | Item 7. MD&A]                │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LangGraph Agent                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ Company         │ Document        │ Retriever       │ Iterative Answerer    │
│ Resolver        │ Fetcher         │ + Query Opt     │ + Quality Evaluator   │
├─────────────────┴─────────────────┴─────────────────┴───────────────────────┤
│                               Components                                     │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ SEC EDGAR       │ Document        │ ChromaDB        │ GLM-4.7 + OpenAI      │
│ Downloader      │ Parser/Chunker  │ VectorStore     │ Embeddings            │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
```

### Quality Refinement Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Generate   │────▶│   Evaluate   │────▶│   Score ≥8?  │
│    Answer    │     │   Quality    │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                          ┌───────────────────────┤
                          │ No                    │ Yes
                          ▼                       ▼
                   ┌──────────────┐        ┌──────────────┐
                   │    Refine    │        │    Return    │
                   │    Answer    │───────▶│    Answer    │
                   └──────────────┘        └──────────────┘
```

## 🛠️ Tech Stack

- **LLM**: GLM-4.7 (Zhipu AI)
- **Embeddings**: OpenAI text-embedding-3-small (default) / GLM embedding-2 (fallback)
- **Framework**: LangChain + LangGraph
- **Vector DB**: ChromaDB (persistent storage)
- **Document Parsing**: BeautifulSoup + Custom HTML→Markdown
- **CLI**: Typer + Rich

## 📁 Supported SEC Forms

| Form Type | Description | Companies |
|-----------|-------------|-----------|
| 10-K | Annual Report | US Domestic |
| 10-Q | Quarterly Report | US Domestic |
| 8-K | Current Report | US Domestic |
| 20-F | Annual Report | Foreign Private Issuers |
| 6-K | Current Report | Foreign Private Issuers |

## 🔗 Integration

### NOVA Integration

AURORA provides utilities for external tools (like NOVA) to check for already downloaded filings, preventing redundant data collection.

See [NOVA_INTEGRATION.md](NOVA_INTEGRATION.md) for details.

**Key Functions:**
- `get_existing_filings_set()`: Get set of already downloaded filings by CIK
- `check_filing_exists_in_aurora()`: Check if a specific filing exists
- `filter_existing_filings()`: Filter out already downloaded filings from a list

## 📜 License

POWERED BY ELCHEMIST
