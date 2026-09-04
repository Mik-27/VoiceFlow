"""Speech-to-text engine interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel


AudioInput = np.ndarray | str | PathLike[str] | bytes
ModelFactory = Callable[..., Any]


class BaseSTTEngine(ABC):
	"""Convert audio input into a transcript."""

	@abstractmethod
	def transcribe(self, audio_data: AudioInput) -> str:
		"""Return the transcript for an array, path, or in-memory PCM buffer."""


class FasterWhisperEngine(BaseSTTEngine):
	"""Transcribe local audio with one long-lived faster-whisper model."""

	def __init__(
		self,
		model_size: str = "small.en",
		device: str = "auto",
		compute_type: str = "auto",
		model_factory: ModelFactory | None = None,
	) -> None:
		if not model_size.strip():
			raise ValueError("model_size must not be empty.")

		resolved_device = self._resolve_device(device)
		resolved_compute_type = self._resolve_compute_type(compute_type, resolved_device)
		factory = model_factory or WhisperModel
		self.model = factory(
			model_size,
			device=resolved_device,
			compute_type=resolved_compute_type,
		)
		self.device = resolved_device
		self.compute_type = resolved_compute_type

	def transcribe(self, audio_data: AudioInput) -> str:
		"""Transcribe 16 kHz audio from a NumPy array, path, or PCM bytes."""
		model_input = self._prepare_input(audio_data)
		segments, _info = self.model.transcribe(
			model_input,
			beam_size=1,
			vad_filter=True,
		)
		return " ".join(
			segment.text.strip()
			for segment in segments
			if getattr(segment, "text", "").strip()
		).strip()

	@staticmethod
	def _resolve_device(device: str) -> str:
		device = device.lower().strip()
		if device == "auto":
			try:
				return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
			except RuntimeError:
				return "cpu"
		if device == "mps":
			return "cpu"
		if device not in {"cuda", "cpu"}:
			raise ValueError("device must be 'auto', 'cuda', 'cpu', or 'mps'.")
		return device

	@staticmethod
	def _resolve_compute_type(compute_type: str, device: str) -> str:
		compute_type = compute_type.lower().strip()
		if compute_type == "auto":
			return "float16" if device == "cuda" else "int8"
		if compute_type not in {"float16", "int8", "int8_float16", "float32"}:
			raise ValueError(
				"compute_type must be 'auto', 'float16', 'int8', 'int8_float16', or 'float32'."
			)
		return compute_type

	@staticmethod
	def _prepare_input(audio_data: AudioInput) -> np.ndarray | str:
		if isinstance(audio_data, np.ndarray):
			if audio_data.ndim > 2 or (audio_data.ndim == 2 and audio_data.shape[1] != 1):
				raise ValueError("audio_data arrays must be one-dimensional mono audio.")
			if audio_data.dtype.kind in {"i", "u"}:
				info = np.iinfo(audio_data.dtype)
				max_value = abs(info.min) if audio_data.dtype.kind == "i" else info.max
				return audio_data.astype(np.float32) / max_value
			return audio_data.astype(np.float32, copy=False)

		if isinstance(audio_data, bytes):
			if len(audio_data) % 2:
				raise ValueError("PCM audio bytes must contain complete int16 samples.")
			return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32_768

		if isinstance(audio_data, (str, PathLike)):
			path = Path(audio_data)
			if not path.exists() or not path.is_file():
				raise FileNotFoundError(f"Audio file does not exist: {path}")
			return str(path)

		raise TypeError("audio_data must be a NumPy array, path, or bytes.")


class MockSTTEngine(BaseSTTEngine):
	"""Deterministic STT engine used until a local model backend is configured."""

	def __init__(self, transcript: str = "This is a mocked transcript.") -> None:
		self._transcript = transcript
		self.last_audio_buffer = b""

	def transcribe(self, audio_data: AudioInput) -> str:
		"""Store test input and return the configured transcript for nonempty audio."""
		if not isinstance(audio_data, bytes):
			raise TypeError("MockSTTEngine expects bytes.")

		self.last_audio_buffer = audio_data
		return self._transcript if audio_data else ""
