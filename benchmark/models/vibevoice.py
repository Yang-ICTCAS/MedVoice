from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .base import BaseModel

_DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR"


class VibeVoiceModel(BaseModel):
    """Microsoft VibeVoice-ASR wrapper via HuggingFace Transformers."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "float32",
    ):
        super().__init__(device)
        self.model_id = model_id
        self.torch_dtype_str = torch_dtype
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "vibevoice-asr"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float32)

        # MPS does not support float16 reliably — fall back to float32
        if self.device == "mps" and torch_dtype == torch.float16:
            warnings.warn(
                "float16 is not fully supported on MPS; switching to float32.",
                RuntimeWarning,
            )
            torch_dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self._model.eval()
        self._loaded = True

    def _transcribe_audio(self, audio_path: Path) -> str:
        import torch
        import soundfile as sf

        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)

        # Resample to 16 kHz if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(self.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(input_features)

        transcription = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return transcription[0] if transcription else ""
