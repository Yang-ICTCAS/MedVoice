from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BaseModel


class WhisperModel(BaseModel):
    """OpenAI Whisper wrapper."""

    def __init__(
        self,
        size: str = "base",
        device: str = "auto",
        language: Optional[str] = None,
        fp16: bool = False,
    ):
        super().__init__(device)
        self.size = size
        self.language = language
        # fp16 only works reliably on CUDA; force off for mps/cpu
        self.fp16 = fp16 and self.device == "cuda"
        self._model = None

    @property
    def name(self) -> str:
        return f"whisper-{self.size}"

    def load(self) -> None:
        import whisper
        self._model = whisper.load_model(self.size, device=self.device)
        self._loaded = True

    def _transcribe_audio(self, audio_path: Path) -> str:
        result = self._model.transcribe(
            str(audio_path),
            language=self.language,
            fp16=self.fp16,
            verbose=False,
        )
        return result["text"]
