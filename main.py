#!/usr/bin/env python3
"""MedVoice Benchmark CLI — compare VibeVoice-ASR and OpenAI Whisper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        console.print(f"[yellow]Config not found at {p}; using defaults.[/]")
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _build_models(cfg: dict, model_filter: list[str] | None = None):
    from benchmark.models.whisper import WhisperModel
    from benchmark.models.vibevoice import VibeVoiceModel

    w_cfg  = cfg.get("models", {}).get("whisper",   {})
    vv_cfg = cfg.get("models", {}).get("vibevoice",  {})

    models = []

    if model_filter is None or "whisper" in model_filter:
        models.append(WhisperModel(
            size    = w_cfg.get("size",   "base"),
            device  = w_cfg.get("device", "auto"),
            language= w_cfg.get("language"),
            fp16    = w_cfg.get("fp16",   False),
        ))

    if model_filter is None or "vibevoice" in model_filter:
        models.append(VibeVoiceModel(
            model_id    = vv_cfg.get("model_id",    "microsoft/VibeVoice-ASR"),
            device      = vv_cfg.get("device",      "auto"),
            torch_dtype = vv_cfg.get("torch_dtype", "float32"),
        ))

    return models


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> None:
    from benchmark.runner import run_benchmark
    from benchmark.reporter import generate_reports

    cfg = _load_config(args.config)
    bm  = cfg.get("benchmark", {})
    rep = cfg.get("report",    {})
    dat = cfg.get("data",      {})

    audio_dir      = Path(args.audio or dat.get("samples_dir",    "data/samples"))
    references_dir = Path(args.references or dat.get("references_dir", "data/references"))
    output_dir     = Path(args.output or bm.get("output_dir", "results"))
    warmup_runs    = args.warmup  if args.warmup  is not None else bm.get("warmup_runs",  1)
    timed_runs     = args.runs    if args.runs    is not None else bm.get("timed_runs",   3)
    formats        = args.formats if args.formats else rep.get("formats", ["console", "json", "html"])
    verbose        = not args.quiet

    model_filter = args.models or None
    models = _build_models(cfg, model_filter)

    if not models:
        console.print("[red]No models selected.[/]")
        sys.exit(1)

    summaries = run_benchmark(
        models         = models,
        audio_dir      = audio_dir,
        references_dir = references_dir,
        warmup_runs    = warmup_runs,
        timed_runs     = timed_runs,
    )

    generate_reports(summaries, output_dir=output_dir, formats=formats, verbose=verbose)


def cmd_run(args: argparse.Namespace) -> None:
    """Transcribe a single audio file with one model and print the result."""
    from benchmark.models.whisper import WhisperModel
    from benchmark.models.vibevoice import VibeVoiceModel

    model_name = args.model.lower()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        console.print(f"[red]Audio file not found: {audio_path}[/]")
        sys.exit(1)

    cfg = _load_config(args.config)

    if model_name == "whisper":
        w_cfg = cfg.get("models", {}).get("whisper", {})
        model = WhisperModel(
            size   = args.size or w_cfg.get("size", "base"),
            device = args.device or w_cfg.get("device", "auto"),
        )
    elif model_name == "vibevoice":
        vv_cfg = cfg.get("models", {}).get("vibevoice", {})
        model = VibeVoiceModel(
            model_id = vv_cfg.get("model_id", "microsoft/VibeVoice-ASR"),
            device   = args.device or vv_cfg.get("device", "auto"),
        )
    else:
        console.print(f"[red]Unknown model: {model_name}. Choose 'whisper' or 'vibevoice'.[/]")
        sys.exit(1)

    result = model.transcribe(audio_path)
    console.print(f"\n[bold]Model:[/]    {result.model_name}")
    console.print(f"[bold]Device:[/]   {result.device}")
    console.print(f"[bold]Latency:[/]  {result.latency_ms:.0f} ms")
    console.print(f"[bold]RTF:[/]      {result.rtf:.4f}")
    console.print(f"[bold]Text:[/]\n{result.text}\n")


def cmd_compare(args: argparse.Namespace) -> None:
    """Pretty-print a side-by-side comparison of two JSON result files."""
    from benchmark.reporter import print_console_report
    from benchmark.metrics import ModelSummary, SampleMetrics

    summaries = []
    for path in args.results:
        data = json.loads(Path(path).read_text())
        for m in data["models"]:
            summary = ModelSummary(model_name=m["model_name"], device=m["device"])
            for s in m["samples"]:
                summary.sample_metrics.append(SampleMetrics(
                    audio_path         = s["audio_path"],
                    audio_duration_s   = s["audio_duration_s"],
                    model_name         = m["model_name"],
                    device             = m["device"],
                    wer                = s.get("wer"),
                    latencies_s        = s.get("latencies_s", [s["mean_latency_s"]]),
                    peak_memories_mb   = [s["mean_peak_memory_mb"]],
                    transcript         = s.get("transcript", ""),
                ))
            summaries.append(summary)

    print_console_report(summaries, verbose=not args.quiet)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medvoice",
        description="Benchmark VibeVoice-ASR vs OpenAI Whisper on macOS",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Run full benchmark")
    p_bm.add_argument("--audio",      help="Directory containing audio files")
    p_bm.add_argument("--references", help="Directory containing reference .txt files")
    p_bm.add_argument("--output",     help="Output directory for reports")
    p_bm.add_argument("--models",     nargs="+", choices=["whisper", "vibevoice"],
                      help="Run only specified models (default: both)")
    p_bm.add_argument("--warmup",     type=int, help="Warm-up runs per sample")
    p_bm.add_argument("--runs",       type=int, help="Timed runs per sample")
    p_bm.add_argument("--formats",    nargs="+", choices=["console", "json", "html"],
                      help="Report output formats")
    p_bm.add_argument("--quiet", "-q", action="store_true", help="Hide per-sample tables")

    # run
    p_run = sub.add_parser("run", help="Transcribe a single audio file")
    p_run.add_argument("--model",  required=True, choices=["whisper", "vibevoice"])
    p_run.add_argument("--audio",  required=True, help="Path to audio file")
    p_run.add_argument("--size",   help="Whisper model size (tiny/base/small/medium/large)")
    p_run.add_argument("--device", help="Device override (cpu/mps/cuda)")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two JSON result files")
    p_cmp.add_argument("results", nargs=2, help="Two JSON result files to compare")
    p_cmp.add_argument("--quiet", "-q", action="store_true", help="Hide per-sample tables")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "benchmark": cmd_benchmark,
        "run":       cmd_run,
        "compare":   cmd_compare,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
