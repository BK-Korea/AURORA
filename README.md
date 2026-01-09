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

AI-powered SEC document analysis with strict citation verification.

## Features

- **Company Resolution**: Fuzzy matching for company names/tickers
- **SEC EDGAR Integration**: Automatic download of 10-K, 10-Q, 8-K filings
- **Table Preservation**: Markdown conversion for financial tables
- **AI-Powered Q&A**: LangChain + LangGraph powered analysis
- **Citation Verification**: Every claim is traced to source documents
- **No Hallucination**: Strict RAG with source-only responses

## Installation

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

## Configuration

Create a `.env` file in the project root:

```env
GLM_API_KEY=your_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

## Usage

### Research a Company

```bash
# Interactive mode
aurora research

# Direct company lookup
aurora research "Apple"
aurora research AAPL --years 5
```

### Ask Questions

```bash
# Interactive Q&A after research
aurora research "Microsoft"

# Quick question on indexed documents
aurora ask "What are the main risk factors?"
```

### Other Commands

```bash
# Check index status
aurora status

# Clear all indexed documents
aurora clear
```

## Example Session

```
$ aurora research

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗          ║
║    ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗         ║
║    ...                                                        ║
╚═══════════════════════════════════════════════════════════════╝

? Enter company name to research: Apple
✓ Found: Apple Inc. (AAPL)
Is this correct? [Y/n]: y
? How many years of filings to download? [3]: 3

Downloading SEC filings (3 years)...
  ▸ Downloaded: 10-K (2024-11-01)
  ▸ Downloaded: 10-Q (2024-08-02)
  ...

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Form Type           ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 10-K                │ 3     │
│ 10-Q                │ 12    │
│ 8-K                 │ 45    │
│ Total Chunks Indexed│ 1,234 │
└─────────────────────┴───────┘

╭─────────────── Q&A Mode ───────────────╮
│ Ready for questions!                   │
│                                        │
│ Ask anything about the company's SEC   │
│ filings. Type quit or exit to end.     │
╰────────────────────────────────────────╯

? Your question: What are Apple's main revenue sources?

╭──────────────── Answer ────────────────╮
│ Apple's main revenue sources are:      │
│                                        │
│ 1. **iPhone**: $200.6B (52% of total)  │
│    [10-K 2024-11-01 | Page 29 | Item 7]│
│                                        │
│ 2. **Services**: $85.2B (22%)          │
│    [10-K 2024-11-01 | Page 29 | Item 7]│
│ ...                                    │
╰────────────────────────────────────────╯
✓ All citations verified
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LangGraph Agent                          │
├───────────────┬───────────────┬─────────────┬──────────────────┤
│ Company       │ Document      │ Retriever   │ Answer Generator │
│ Resolver      │ Fetcher       │ Node        │ + Citation Valid │
├───────────────┴───────────────┴─────────────┴──────────────────┤
│                           Components                             │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│ SEC         │ Document    │ Chroma      │ GLM-4.7             │
│ Downloader  │ Parser      │ VectorStore │ LLM Client          │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

## Tech Stack

- **LLM**: GLM-4.7 (Zhipu AI)
- **Embeddings**: embedding-3 (Zhipu AI)
- **Framework**: LangChain + LangGraph
- **Vector DB**: Chroma
- **Document Parsing**: BeautifulSoup + Custom HTML→Markdown
- **CLI**: Typer + Rich

## License

MIT License
