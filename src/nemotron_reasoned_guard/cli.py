"""Simple CLI for nemotron-reasoned-guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .guard import ReasonedGuard
from .policies import DEFAULT_POLICIES, load_policy

app = typer.Typer(
    name="nemotron-guard",
    help="Policy-as-code guardrails using NVIDIA Nemotron reasoned safety models.",
    add_completion=False,
)
console = Console()


@app.command()
def check(
    text: str = typer.Argument(..., help="Text content to evaluate"),
    policy: str = typer.Option(
        "corporate",
        "--policy",
        "-p",
        help="Built-in policy name or path to a .yaml policy file",
    ),
    image: Optional[str] = typer.Option(None, "--image", "-i", help="Path or URL to image"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Extra context"),
    temperature: float = typer.Option(0.2, "--temp", help="Sampling temperature"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output full result as JSON"),
):
    """Run a single reasoned guard check."""
    guard = ReasonedGuard()

    if Path(policy).exists():
        pol = load_policy(policy)
    else:
        pol = DEFAULT_POLICIES.get(policy, DEFAULT_POLICIES["corporate"])

    result = guard.check(
        text=text,
        policy=pol,
        image=image,
        context=context,
        temperature=temperature,
    )

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
        return

    verdict = "[green]SAFE[/green]" if result.is_safe else "[red]UNSAFE[/red]"
    console.print(Panel(f"Verdict: {verdict}", title="Nemotron Reasoned Guard"))

    console.print(f"Policy: [bold]{result.policy_name}[/bold]")
    if result.categories:
        console.print(f"Categories: {', '.join(result.categories)}")
    if result.confidence is not None:
        console.print(f"Confidence: {result.confidence:.2f}")

    console.print("\n[bold]Reasoning trace:[/bold]")
    console.print(Panel(result.reasoning, border_style="blue"))


@app.command()
def list_policies():
    """List available built-in policies."""
    console.print("[bold]Built-in policies:[/bold]")
    for name, pol in DEFAULT_POLICIES.items():
        console.print(f"  • [cyan]{name}[/cyan] — {pol.description}")


@app.command()
def playground():
    """Launch the Gradio playground."""
    try:
        from playground.app import build_ui
    except ImportError:
        from ..playground.app import build_ui  # type: ignore

    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    app()
