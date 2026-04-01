from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .metrics import ModelSummary, SampleMetrics, aggregate_results
from .models.base import BaseModel, TranscribeResult

logger = logging.getLogger(__name__)
console = Console()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _collect_audio_files(audio_dir: Path) -> List[Path]:
    files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def _load_references(references_dir: Path) -> Dict[str, str]:
    """Return {audio_path_stem: reference_text} mapping."""
    ref_map: Dict[str, str] = {}
    if not references_dir.exists():
        return ref_map
    for ref_file in references_dir.glob("*.txt"):
        text = ref_file.read_text(encoding="utf-8").strip()
        ref_map[ref_file.stem] = text
    return ref_map


def run_benchmark(
    models: List[BaseModel],
    audio_dir: Path,
    references_dir: Path,
    warmup_runs: int = 1,
    timed_runs: int = 3,
    single_audio: Optional[Path] = None,
) -> List[ModelSummary]:
    """
    Run all models over all audio files.
    Returns a ModelSummary per model.
    """
    if single_audio:
        audio_files = [single_audio]
    else:
        audio_files = _collect_audio_files(audio_dir)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    ref_by_stem = _load_references(references_dir)
    # Build a path-keyed reference map
    reference_map: Dict[str, str] = {}
    for af in audio_files:
        if af.stem in ref_by_stem:
            reference_map[str(af)] = ref_by_stem[af.stem]

    summaries: List[ModelSummary] = []

    for model in models:
        console.rule(f"[bold cyan]{model.name}[/] on [yellow]{model.device}[/]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task(f"Loading {model.name}...", total=None)
            model.load()
            progress.update(load_task, completed=True, description=f"{model.name} loaded")

        summary = ModelSummary(model_name=model.name, device=model.device)

        for af in audio_files:
            console.print(f"  [dim]Audio:[/] {af.name}")

            # Warm-up runs (not timed)
            for _ in range(warmup_runs):
                try:
                    model.transcribe(af)
                except Exception as e:
                    logger.warning("Warm-up failed for %s / %s: %s", model.name, af.name, e)

            # Timed runs
            timed_results: List[TranscribeResult] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                run_task = progress.add_task(
                    f"    Timing ({timed_runs} runs)...", total=timed_runs
                )
                for i in range(timed_runs):
                    try:
                        result = model.transcribe(af)
                        timed_results.append(result)
                    except Exception as e:
                        logger.error("Run %d failed for %s / %s: %s", i + 1, model.name, af.name, e)
                    progress.advance(run_task)

            if not timed_results:
                console.print(f"    [red]All runs failed for {af.name}[/]")
                continue

            sample_metrics = aggregate_results(timed_results, reference_map)
            summary.sample_metrics.append(sample_metrics)

            wer_str = f"{sample_metrics.wer:.1%}" if sample_metrics.wer is not None else "n/a"
            console.print(
                f"    latency={sample_metrics.mean_latency_s:.2f}s  "
                f"RTF={sample_metrics.mean_rtf:.3f}  "
                f"WER={wer_str}  "
                f"mem={sample_metrics.mean_peak_memory_mb:.1f}MB"
            )

        summaries.append(summary)

    return summaries
