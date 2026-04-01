from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich import box

from .metrics import ModelSummary, SampleMetrics

console = Console()


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _fmt_wer(wer: Optional[float]) -> str:
    return f"{wer:.1%}" if wer is not None else "n/a"


def _winner_style(values: list, lower_is_better: bool = True) -> list[str]:
    """Return rich style strings; highlight the best value green."""
    if len(values) < 2 or any(v is None for v in values):
        return [""] * len(values)
    best = min(values) if lower_is_better else max(values)
    return ["bold green" if v == best else "" for v in values]


def print_console_report(summaries: List[ModelSummary], verbose: bool = True) -> None:
    console.rule("[bold white]Benchmark Results[/]")

    # ---- Summary table ----
    summary_table = Table(
        title="Model Summary",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Model", style="cyan", no_wrap=True)
    summary_table.add_column("Device")
    summary_table.add_column("Mean WER ↓", justify="right")
    summary_table.add_column("Mean Latency (s) ↓", justify="right")
    summary_table.add_column("Mean RTF ↓", justify="right")
    summary_table.add_column("Mean Peak Mem (MB) ↓", justify="right")

    wers   = [s.mean_wer         for s in summaries]
    lats   = [s.mean_latency_s   for s in summaries]
    rtfs   = [s.mean_rtf         for s in summaries]
    mems   = [s.mean_peak_memory_mb for s in summaries]

    wer_styles = _winner_style(wers)
    lat_styles = _winner_style(lats)
    rtf_styles = _winner_style(rtfs)
    mem_styles = _winner_style(mems)

    for i, s in enumerate(summaries):
        summary_table.add_row(
            s.model_name,
            s.device,
            f"[{wer_styles[i]}]{_fmt_wer(s.mean_wer)}[/]",
            f"[{lat_styles[i]}]{s.mean_latency_s:.3f}[/]",
            f"[{rtf_styles[i]}]{s.mean_rtf:.4f}[/]",
            f"[{mem_styles[i]}]{s.mean_peak_memory_mb:.1f}[/]",
        )

    console.print(summary_table)

    # ---- Per-sample table ----
    if verbose:
        for s in summaries:
            sample_table = Table(
                title=f"{s.model_name} — per-sample breakdown",
                box=box.SIMPLE_HEAVY,
                header_style="bold blue",
            )
            sample_table.add_column("Audio file", style="dim")
            sample_table.add_column("Duration (s)", justify="right")
            sample_table.add_column("WER ↓", justify="right")
            sample_table.add_column("Latency (s) ↓", justify="right")
            sample_table.add_column("RTF ↓", justify="right")
            sample_table.add_column("Peak Mem (MB)", justify="right")
            sample_table.add_column("Transcript", max_width=50, no_wrap=False)

            for sm in s.sample_metrics:
                sample_table.add_row(
                    Path(sm.audio_path).name,
                    f"{sm.audio_duration_s:.1f}",
                    _fmt_wer(sm.wer),
                    f"{sm.mean_latency_s:.3f}",
                    f"{sm.mean_rtf:.4f}",
                    f"{sm.mean_peak_memory_mb:.1f}",
                    sm.transcript[:120] + ("…" if len(sm.transcript) > 120 else ""),
                )

            console.print(sample_table)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def _summary_to_dict(s: ModelSummary) -> dict:
    return {
        "model_name": s.model_name,
        "device": s.device,
        "mean_wer": s.mean_wer,
        "mean_latency_s": s.mean_latency_s,
        "mean_rtf": s.mean_rtf,
        "mean_peak_memory_mb": s.mean_peak_memory_mb,
        "samples": [
            {
                "audio_path": sm.audio_path,
                "audio_duration_s": sm.audio_duration_s,
                "wer": sm.wer,
                "mean_latency_s": sm.mean_latency_s,
                "median_latency_s": sm.median_latency_s,
                "mean_rtf": sm.mean_rtf,
                "mean_peak_memory_mb": sm.mean_peak_memory_mb,
                "latencies_s": sm.latencies_s,
                "transcript": sm.transcript,
            }
            for sm in s.sample_metrics
        ],
    }


def save_json_report(summaries: List[ModelSummary], output_path: Path) -> None:
    data = {
        "generated_at": datetime.now().isoformat(),
        "models": [_summary_to_dict(s) for s in summaries],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]JSON report saved:[/] {output_path}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def save_html_report(summaries: List[ModelSummary], output_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        console.print("[yellow]plotly not installed; skipping HTML report.[/]")
        return

    model_names = [s.model_name for s in summaries]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Mean WER (lower is better)",
                        "Mean Latency / s (lower is better)",
                        "Real-Time Factor (lower is better)",
                        "Peak Memory / MB (lower is better)"],
    )

    wers  = [s.mean_wer * 100 if s.mean_wer is not None else 0 for s in summaries]
    lats  = [s.mean_latency_s  for s in summaries]
    rtfs  = [s.mean_rtf        for s in summaries]
    mems  = [s.mean_peak_memory_mb for s in summaries]

    bar_kwargs = dict(marker_color=colors[:len(summaries)])

    fig.add_trace(go.Bar(x=model_names, y=wers,  name="WER (%)",    **bar_kwargs), row=1, col=1)
    fig.add_trace(go.Bar(x=model_names, y=lats,  name="Latency (s)", **bar_kwargs), row=1, col=2)
    fig.add_trace(go.Bar(x=model_names, y=rtfs,  name="RTF",         **bar_kwargs), row=2, col=1)
    fig.add_trace(go.Bar(x=model_names, y=mems,  name="Mem (MB)",    **bar_kwargs), row=2, col=2)

    fig.update_layout(
        title_text="ASR Benchmark: VibeVoice vs Whisper",
        showlegend=False,
        height=700,
        template="plotly_dark",
    )

    # Per-sample latency scatter
    scatter_fig = go.Figure()
    for i, s in enumerate(summaries):
        for sm in s.sample_metrics:
            scatter_fig.add_trace(go.Box(
                y=sm.latencies_s,
                name=f"{s.model_name}\n{Path(sm.audio_path).name}",
                marker_color=colors[i % len(colors)],
            ))
    scatter_fig.update_layout(
        title="Latency distribution per sample",
        yaxis_title="Latency (s)",
        template="plotly_dark",
        height=400,
    )

    bar_html    = pio.to_html(fig,         full_html=False, include_plotlyjs="cdn")
    scatter_html = pio.to_html(scatter_fig, full_html=False, include_plotlyjs=False)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MedVoice ASR Benchmark</title>
  <style>
    body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 2rem; }}
    h1   {{ color: #a29bfe; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #444; padding: .5rem 1rem; text-align: right; }}
    th   {{ background: #2d2d6b; }}
    .best {{ color: #00cc96; font-weight: bold; }}
    .meta {{ color: #888; font-size: .85rem; }}
  </style>
</head>
<body>
<h1>MedVoice ASR Benchmark Report</h1>
<p class="meta">Generated: {timestamp}</p>

<h2>Summary</h2>
<table>
  <tr>
    <th>Model</th><th>Device</th>
    <th>Mean WER ↓</th><th>Mean Latency (s) ↓</th>
    <th>Mean RTF ↓</th><th>Peak Mem (MB) ↓</th>
  </tr>
{"".join(
    f"<tr><td>{s.model_name}</td><td>{s.device}</td>"
    f"<td>{_fmt_wer(s.mean_wer)}</td>"
    f"<td>{s.mean_latency_s:.3f}</td>"
    f"<td>{s.mean_rtf:.4f}</td>"
    f"<td>{s.mean_peak_memory_mb:.1f}</td></tr>"
    for s in summaries
)}
</table>

{bar_html}
{scatter_html}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    console.print(f"[green]HTML report saved:[/] {output_path}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def generate_reports(
    summaries: List[ModelSummary],
    output_dir: Path,
    formats: List[str],
    verbose: bool = True,
    run_id: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "console" in formats:
        print_console_report(summaries, verbose=verbose)
    if "json" in formats:
        save_json_report(summaries, output_dir / f"benchmark_{run_id}.json")
    if "html" in formats:
        save_html_report(summaries, output_dir / f"benchmark_{run_id}.html")
