"""AURORA - SEC Company Research Tool CLI."""
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import AuroraAgent
from src.config import get_settings

# Load environment variables
load_dotenv()

app = typer.Typer(help="AURORA - SEC Company Research Tool")
console = Console()


AURORA_BANNER = """
[bold cyan]
                ╔═══════════════════════════════════════════════════════════════╗
                ║                                                               ║
                ║     █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗          ║
                ║    ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗         ║
                ║    ███████║██║   ██║██████╔╝██║   ██║██████╔╝███████║         ║
                ║    ██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══██║         ║
                ║    ██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║  ██║         ║
                ║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝         ║
                ║                                                               ║
                ║         [bold white]◈ Autonomous Research & Analysis System ◈[/bold white]          ║
                ║                                                               ║
                ╠═══════════════════════════════════════════════════════════════╣
                ║                                                               ║
                ║    [dim]▸ SEC EDGAR Integration[/dim]     [dim]▸ AI-Powered Analysis[/dim]         ║
                ║    [dim]▸ Multi-Agent Team[/dim]          [dim]▸ Risk Assessment[/dim]            ║
                ║    [dim]▸ Citation Verification[/dim]     [dim]▸ Executive Reports[/dim]          ║
                ║                                                               ║
                ╚═══════════════════════════════════════════════════════════════╝
[/bold cyan]
"""

AURORA_SUBTITLE = """
[dim italic]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    Illuminating Corporate Intelligence from SEC Filings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim italic]
"""


def display_banner():
    """Display the AURORA startup banner."""
    console.print(AURORA_BANNER)
    console.print(AURORA_SUBTITLE)
    console.print()


def create_agent() -> AuroraAgent:
    """Create and configure the AURORA agent."""
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        console.print("[red]Error: GLM_API_KEY not found in environment variables.[/red]")
        console.print("[dim]Please set your API key in .env file[/dim]")
        raise typer.Exit(1)

    # Check for placeholder values
    if api_key in ["your_api_key_here", "your-api-key-here", "YOUR_API_KEY_HERE"]:
        console.print("[red]Error: GLM_API_KEY contains a placeholder value.[/red]")
        console.print("[dim]Please update .env file with your actual Zhipu AI API key[/dim]")
        console.print("[dim]Get your key at: https://open.bigmodel.cn/[/dim]")
        raise typer.Exit(1)

    # OpenAI API key for embeddings (optional but recommended)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_provider = "openai" if openai_api_key else "glm"

    if openai_api_key:
        console.print("[dim]Using OpenAI for embeddings[/dim]")
    else:
        console.print("[dim]Using GLM for embeddings (set OPENAI_API_KEY for better results)[/dim]")

    base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    data_dir = Path(__file__).parent.parent / "data"

    def progress_callback(message: str):
        console.print(f"  [dim]▸[/dim] {message}")

    return AuroraAgent(
        api_key=api_key,
        base_url=base_url,
        data_dir=data_dir,
        progress_callback=progress_callback,
        openai_api_key=openai_api_key,
        embedding_provider=embedding_provider,
    )


def _display_team_result(result: dict):
    """Display team analysis result with professional formatting."""
    report = result.get("team_report") or result.get("current_answer", "")
    risk_rating = result.get("risk_rating", "N/A")
    risk_flags = result.get("risk_flags", [])
    confidence = result.get("analysis_confidence", 0.0)
    analysts_used = result.get("analysts_used", [])
    key_metrics = result.get("key_metrics", {})

    # Analyst display names
    analyst_names = {
        "financial_analyst": "Financial",
        "risk_analyst": "Risk",
        "comparative_analyst": "Comparative",
    }
    analysts_display = ", ".join(
        analyst_names.get(a, a) for a in analysts_used
    ) or "N/A"

    # Risk rating color
    risk_colors = {
        "LOW": "green",
        "MODERATE": "yellow",
        "ELEVATED": "dark_orange",
        "HIGH": "red",
        "CRITICAL": "bold red",
    }
    risk_color = risk_colors.get(risk_rating, "white")

    header_text = (
        f"[bold]Analysts[/bold]: {analysts_display}  |  "
        f"[bold]Risk[/bold]: [{risk_color}]{risk_rating}[/{risk_color}]  |  "
        f"[bold]Confidence[/bold]: {confidence:.0%}"
    )

    console.print()
    console.print(Panel(
        header_text,
        title="[bold cyan]AURORA Team Analysis[/bold cyan]",
        border_style="cyan",
        padding=(0, 2),
    ))

    # Key metrics dashboard
    financial_health = key_metrics.get("financial_health_score")
    agg_risk = key_metrics.get("aggregate_risk_score")
    trajectory = key_metrics.get("overall_trajectory")
    total_risks = key_metrics.get("total_risks_identified")

    if any([financial_health, agg_risk, trajectory, total_risks]):
        metrics_table = Table(
            title="Key Indicators",
            show_header=True,
            border_style="dim",
            padding=(0, 1),
        )
        metrics_table.add_column("Indicator", style="cyan", min_width=20)
        metrics_table.add_column("Value", style="bold", min_width=15)
        metrics_table.add_column("Status", min_width=10)

        if financial_health is not None:
            fh = int(financial_health) if isinstance(financial_health, (int, float)) else 0
            fh_status = (
                "[green]Healthy[/green]" if fh >= 7
                else "[yellow]Fair[/yellow]" if fh >= 5
                else "[red]Weak[/red]"
            )
            metrics_table.add_row("Financial Health", f"{fh}/10", fh_status)

        if agg_risk is not None:
            ar = float(agg_risk) if isinstance(agg_risk, (int, float)) else 0
            ar_status = (
                "[green]Low[/green]" if ar <= 3
                else "[yellow]Moderate[/yellow]" if ar <= 6
                else "[red]High[/red]"
            )
            metrics_table.add_row("Risk Score", f"{ar:.1f}/10", ar_status)

        if trajectory:
            traj_status = (
                "[green]Positive[/green]" if "improv" in str(trajectory).lower()
                else "[red]Negative[/red]" if "deterior" in str(trajectory).lower()
                else "[yellow]Stable[/yellow]"
            )
            metrics_table.add_row("Trajectory", str(trajectory), traj_status)

        if total_risks is not None:
            high_risks = key_metrics.get("high_severity_risks", 0)
            risk_status = (
                "[red]Alert[/red]" if high_risks >= 3
                else "[yellow]Monitor[/yellow]" if high_risks >= 1
                else "[green]OK[/green]"
            )
            metrics_table.add_row(
                "Risks Identified",
                f"{total_risks} (High: {high_risks})",
                risk_status,
            )

        console.print()
        console.print(metrics_table)

    # Risk flags
    if risk_flags:
        console.print()
        risk_panel_lines = []
        for flag in risk_flags[:5]:
            risk_panel_lines.append(f"[red]![/red] {flag}")
        console.print(Panel(
            "\n".join(risk_panel_lines),
            title="[bold red]Risk Alerts[/bold red]",
            border_style="red",
            padding=(0, 2),
        ))

    # Main report
    if report:
        console.print()
        console.print(Panel(
            Markdown(report),
            title="[bold green]Executive Research Report[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))


@app.command()
def research(
    company: Optional[str] = typer.Argument(None, help="Company name or ticker to research"),
    years: int = typer.Option(3, "--years", "-y", help="Years of filings to download"),
):
    """Start a research session on a company."""
    display_banner()

    # Create agent
    with console.status("[bold cyan]Initializing AURORA System...[/bold cyan]"):
        try:
            agent = create_agent()
        except Exception as e:
            console.print(f"[red]Failed to initialize: {e}[/red]")
            raise typer.Exit(1)

    console.print("[green]System initialized[/green]\n")

    # Get company name if not provided
    if not company:
        company = Prompt.ask("[bold cyan]?[/bold cyan] Enter company name to research")

    if not company:
        console.print("[red]No company specified. Exiting.[/red]")
        raise typer.Exit(1)

    # Resolve company
    console.print(f"\n[cyan]Searching for:[/cyan] {company}")

    with console.status("[bold cyan]Resolving company...[/bold cyan]"):
        result = agent.resolve_company(company)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    company_info = result.get("company_info")
    if not company_info:
        console.print("[red]Could not find company.[/red]")
        raise typer.Exit(1)

    # Confirm company
    console.print(f"\n[green]Found:[/green] [bold]{company_info.name}[/bold] ({company_info.ticker})")

    if not Confirm.ask("Is this correct?", default=True):
        console.print("[yellow]Please try a more specific company name.[/yellow]")
        raise typer.Exit(0)

    # Ask for years
    years = int(Prompt.ask(
        "[bold cyan]?[/bold cyan] How many years of filings to download?",
        default=str(years)
    ))

    # Download documents
    console.print(f"\n[cyan]Downloading SEC filings ({years} years)...[/cyan]\n")

    result = agent.fetch_documents(company, years=years)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    filings = result.get("filings", [])
    chunks = result.get("chunks_indexed", 0)

    # Display summary
    summary_table = Table(title="Download Summary", show_header=True)
    summary_table.add_column("Form Type", style="cyan")
    summary_table.add_column("Count", style="green")

    form_counts = {}
    for f in filings:
        form_counts[f.form_type] = form_counts.get(f.form_type, 0) + 1

    for form_type, count in sorted(form_counts.items()):
        summary_table.add_row(form_type, str(count))

    summary_table.add_row("[bold]Total Chunks Indexed[/bold]", f"[bold]{chunks}[/bold]")

    console.print()
    console.print(summary_table)
    console.print()

    # Enter Q&A mode
    console.print(Panel(
        "[bold]Ready for questions![/bold]\n\n"
        "Commands:\n"
        "  [bold cyan]team <question>[/bold cyan] - Full team analysis (Financial + Risk + Comparative)\n"
        "  [bold cyan]<question>[/bold cyan]      - Quick single-agent answer\n"
        "  [bold cyan]quit[/bold cyan]             - Exit session",
        title="Q&A Mode",
        border_style="cyan"
    ))

    while True:
        console.print()
        question = Prompt.ask("[bold cyan]?[/bold cyan] Your question")

        if question.lower() in ["quit", "exit", "q"]:
            console.print("\n[dim]Thank you for using AURORA. Goodbye![/dim]\n")
            break

        if not question.strip():
            continue

        # Check for team analysis mode
        is_team_mode = question.lower().startswith("team ")
        if is_team_mode:
            question = question[5:].strip()
            if not question:
                console.print("[yellow]Please provide a question after 'team'.[/yellow]")
                continue

            result = agent.team_analyze(question)

            if result.get("error") and not (result.get("team_report") or result.get("current_answer")):
                console.print(f"[yellow]Note: {result['error']}[/yellow]")
                continue

            if result.get("error"):
                console.print(f"[yellow]Warning: {result['error']}[/yellow]")

            _display_team_result(result)
        else:
            # Standard mode
            result = agent.ask(question)

            if result.get("error"):
                console.print(f"[yellow]Note: {result['error']}[/yellow]")

            answer = result.get("current_answer", "")
            if answer:
                console.print()
                console.print(Panel(
                    Markdown(answer),
                    title="[bold green]Answer[/bold green]",
                    border_style="green",
                    padding=(1, 2)
                ))

                # Citation validation status
                if result.get("citations_valid"):
                    console.print("[dim green]All citations verified[/dim green]")
                elif result.get("citations_valid") is False:
                    console.print(
                        f"[dim yellow]Warning: "
                        f"{result.get('error', 'Some citations could not be verified')}"
                        f"[/dim yellow]"
                    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about indexed documents"),
):
    """Ask a question about previously indexed documents."""
    console.print()

    with console.status("[bold cyan]Initializing...[/bold cyan]"):
        agent = create_agent()

    stats = agent.get_stats()
    if stats.get("total_chunks", 0) == 0:
        console.print("[yellow]No documents indexed. Run 'aurora research <company>' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[dim]Using {stats['total_chunks']} indexed chunks[/dim]")

    # Show indexed companies if available
    indexed_companies = stats.get("indexed_companies", [])
    if indexed_companies:
        console.print(f"[dim]Indexed companies: {', '.join(indexed_companies[:3])}[/dim]")
    console.print()

    # Don't use status spinner - let progress messages show
    result = agent.ask(question)

    # Check for errors
    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")

        # Show matched company info if available
        matched_company = result.get("matched_company")
        if matched_company:
            console.print(
                f"\n[dim]Matched company: {matched_company.name} ({matched_company.ticker})[/dim]"
            )

        raise typer.Exit(1)

    # Show matched company info if available
    matched_company = result.get("matched_company")
    if matched_company:
        console.print(
            f"[dim green]Using company: {matched_company.name} ({matched_company.ticker})[/dim green]\n"
        )

    answer = result.get("current_answer", "")
    score = result.get("answer_score", 0)

    if answer:
        console.print()
        console.print(Panel(
            Markdown(answer),
            title=f"[bold green]Answer[/bold green] [dim](Quality: {score}/10)[/dim]",
            border_style="green",
            padding=(1, 2)
        ))
    else:
        console.print("[yellow]No answer generated. Please try a different question.[/yellow]")


@app.command(name="team-analyze")
def team_analyze(
    question: str = typer.Argument(..., help="Question for team analysis"),
):
    """
    Run multi-agent team analysis on indexed documents.

    Deploys 3 specialist analysts in parallel:
    - Financial Analyst: Metrics, ratios, financial health
    - Risk Analyst: Risk identification, scoring, going concern detection
    - Comparative Analyst: Cross-period trends, trajectory, management guidance

    Results are synthesized into an executive research report.
    """
    display_banner()

    with console.status("[bold cyan]Initializing AURORA Team...[/bold cyan]"):
        agent = create_agent()

    stats = agent.get_stats()
    if stats.get("total_chunks", 0) == 0:
        console.print("[yellow]No documents indexed. Run 'aurora research <company>' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[dim]Using {stats['total_chunks']} indexed chunks[/dim]")

    indexed_companies = stats.get("indexed_companies", [])
    if indexed_companies:
        console.print(f"[dim]Indexed companies: {', '.join(indexed_companies[:5])}[/dim]")
    console.print()

    # Run team analysis
    result = agent.team_analyze(question)

    if result.get("error") and not (result.get("team_report") or result.get("current_answer")):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    if result.get("error"):
        console.print(f"[yellow]Warning: {result['error']}[/yellow]")

    _display_team_result(result)


@app.command()
def status():
    """Show current index status."""
    display_banner()

    with console.status("[bold cyan]Loading...[/bold cyan]"):
        agent = create_agent()
        stats = agent.get_stats()

    table = Table(title="AURORA Status", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks Indexed", str(stats.get("total_chunks", 0)))
    table.add_row("Collection Name", stats.get("collection_name", "N/A"))
    table.add_row("Storage Path", stats.get("persist_dir", "N/A"))

    indexed_companies = stats.get("indexed_companies", [])
    if indexed_companies:
        table.add_row("Indexed Companies", ", ".join(indexed_companies))

    indexed_tickers = stats.get("indexed_tickers", {})
    if indexed_tickers:
        ticker_list = [f"{name}: {ticker}" for name, ticker in indexed_tickers.items()]
        table.add_row("Tickers", ", ".join(ticker_list))

    earliest = stats.get("earliest_filing_date")
    latest = stats.get("latest_filing_date")
    if earliest and latest:
        table.add_row("Filing Date Range", f"{earliest} ~ {latest}")

    table.add_row("Analysis Modes", "Standard (single-agent), Team (multi-agent)")

    console.print(table)


@app.command()
def clear():
    """Clear all indexed documents."""
    if Confirm.ask("[yellow]Are you sure you want to clear all indexed documents?[/yellow]"):
        with console.status("[bold cyan]Clearing...[/bold cyan]"):
            agent = create_agent()
            agent.vector_store.clear_collection()
        console.print("[green]All documents cleared.[/green]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
