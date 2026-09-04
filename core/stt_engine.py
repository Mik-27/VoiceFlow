"""Speech-to-text engine interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSTTEngine(ABC):
	"""Convert in-memory signed 16-bit PCM audio into a transcript."""

	@abstractmethod
	def transcribe(self, audio_buffer: bytes) -> str:
		"""Return the transcript for an in-memory PCM audio buffer."""


class MockSTTEngine(BaseSTTEngine):
	"""Deterministic STT engine used until a local model backend is configured."""

	def __init__(self, transcript: str = "This is a mocked transcript.") -> None:
		self._transcript = transcript
		self.last_audio_buffer = b""

	def transcribe(self, audio_buffer: bytes) -> str:
		"""Store test input and return the configured transcript for nonempty audio."""
		if not isinstance(audio_buffer, bytes):
			raise TypeError("audio_buffer must be bytes.")

		self.last_audio_buffer = audio_buffer
		return self._transcript if audio_buffer else ""
