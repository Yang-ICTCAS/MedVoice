from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from .models.base import TranscribeResult


@dataclass
class SampleMetrics:
    """Aggregated metrics for one audio sample across multiple runs."""
    audio_path: str
    audio_duration_s: float
    model_name: str
    device: str
    # WER in [0, 1]; None when no reference transcript is available
    wer: Optional[float]
    # Latency across timed runs (seconds)
    latencies_s: List[float]
    # Peak memory across timed runs (MB)
    peak_memories_mb: List[float]
    transcript: str

    @property
    def mean_latency_s(self) -> float:
        return statistics.mean(self.latencies_s) if self.latencies_s else 0.0

    @property
    def median_latency_s(self) -> float:
        return statistics.median(self.latencies_s) if self.latencies_s else 0.0

    @property
    def mean_rtf(self) -> float:
        if self.audio_duration_s == 0:
            return float("inf")
        return self.mean_latency_s / self.audio_duration_s

    @property
    def mean_peak_memory_mb(self) -> float:
        return statistics.mean(self.peak_memories_mb) if self.peak_memories_mb else 0.0


@dataclass
class ModelSummary:
    """Aggregated benchmark summary for one model across all samples."""
    model_name: str
    device: str
    sample_metrics: List[SampleMetrics] = field(default_factory=list)

    @property
    def mean_wer(self) -> Optional[float]:
        wers = [s.wer for s in self.sample_metrics if s.wer is not None]
        return statistics.mean(wers) if wers else None

    @property
    def mean_latency_s(self) -> float:
        lats = [s.mean_latency_s for s in self.sample_metrics]
        return statistics.mean(lats) if lats else 0.0

    @property
    def mean_rtf(self) -> float:
        rtfs = [s.mean_rtf for s in self.sample_metrics]
        return statistics.mean(rtfs) if rtfs else 0.0

    @property
    def mean_peak_memory_mb(self) -> float:
        mems = [s.mean_peak_memory_mb for s in self.sample_metrics]
        return statistics.mean(mems) if mems else 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """Return Word Error Rate in [0, 1] using jiwer."""
    try:
        from jiwer import wer
        return float(wer(reference, hypothesis))
    except ImportError:
        return _wer_fallback(reference, hypothesis)


def _wer_fallback(reference: str, hypothesis: str) -> float:
    """Levenshtein-based WER without jiwer dependency."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m] / n


def aggregate_results(
    results: List[TranscribeResult],
    reference_map: dict[str, str],
) -> SampleMetrics:
    """Collapse multiple runs of the same (model, audio) into one SampleMetrics."""
    assert results, "results must be non-empty"
    first = results[0]
    ref = reference_map.get(first.audio_path)
    wer_val = compute_wer(ref, first.text) if ref else None

    return SampleMetrics(
        audio_path=first.audio_path,
        audio_duration_s=first.audio_duration_s,
        model_name=first.model_name,
        device=first.device,
        wer=wer_val,
        latencies_s=[r.latency_s for r in results],
        peak_memories_mb=[r.peak_memory_mb for r in results],
        transcript=first.text,
    )
