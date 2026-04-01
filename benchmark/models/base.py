from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TranscribeResult:
    """Output of a single transcription call."""
    text: str
    # Wall-clock time for the transcription call (seconds)
    latency_s: float
    # Duration of the audio file (seconds)
    audio_duration_s: float
    # Peak memory delta during transcription (MB)
    peak_memory_mb: float
    # Device used (cpu / mps / cuda)
    device: str
    # Model name/identifier
    model_name: str
    # Path to the source audio
    audio_path: str

    @property
    def rtf(self) -> float:
        """Real-Time Factor: latency / audio_duration. <1.0 means faster than real-time."""
        if self.audio_duration_s == 0:
            return float("inf")
        return self.latency_s / self.audio_duration_s

    @property
    def latency_ms(self) -> float:
        return self.latency_s * 1000


class BaseModel(ABC):
    """Common interface every ASR model wrapper must implement."""

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._loaded = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def load(self) -> None:
        """Download/initialise the model. Called once before any transcription."""

    @abstractmethod
    def _transcribe_audio(self, audio_path: Path) -> str:
        """Return raw transcript text for the given audio file."""

    def transcribe(self, audio_path: str | Path) -> TranscribeResult:
        """Timed transcription with memory profiling."""
        if not self._loaded:
            self.load()

        audio_path = Path(audio_path)
        audio_duration_s = self._get_audio_duration(audio_path)

        tracemalloc.start()
        t0 = time.perf_counter()
        text = self._transcribe_audio(audio_path)
        latency_s = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return TranscribeResult(
            text=text.strip(),
            latency_s=latency_s,
            audio_duration_s=audio_duration_s,
            peak_memory_mb=peak / 1024 / 1024,
            device=self.device,
            model_name=self.name,
            audio_path=str(audio_path),
        )

    @staticmethod
    def _get_audio_duration(audio_path: Path) -> float:
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            try:
                import librosa
                duration = librosa.get_duration(path=str(audio_path))
                return duration
            except Exception:
                return 0.0
