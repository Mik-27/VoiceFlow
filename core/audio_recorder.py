"""Microphone audio capture and buffer management."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

import sounddevice as sd

from config import Settings
from utils.logger import get_logger


class AudioRecorderError(RuntimeError):
	"""Raised when microphone capture cannot be started or used."""


StreamFactory = Callable[..., Any]


class AudioRecorder:
	"""Capture microphone audio as in-memory signed 16-bit PCM bytes."""

	_SAMPLE_WIDTH_BYTES = 2

	def __init__(self, settings: Settings, stream_factory: StreamFactory | None = None) -> None:
		if settings.channels != 1:
			raise ValueError("AudioRecorder supports mono capture only.")

		self._sample_rate = settings.sample_rate
		self._channels = settings.channels
		self._block_size = max(1, round(settings.sample_rate * settings.audio_block_duration_ms / 1_000))
		self._max_buffer_size = (
			settings.sample_rate
			* settings.channels
			* self._SAMPLE_WIDTH_BYTES
			* settings.max_recording_duration_seconds
		)
		self._stream_factory = stream_factory or sd.RawInputStream
		self._buffer = bytearray()
		self._chunk_callbacks: list[Callable[[bytes], None]] = []
		self._stream: Any | None = None
		self._recording = False
		self._lock = threading.RLock()
		self._logger = get_logger("audio_recorder")

	@property
	def is_recording(self) -> bool:
		"""Return whether the input stream is accepting audio samples."""
		with self._lock:
			return self._recording

	@property
	def sample_rate(self) -> int:
		"""Return the captured PCM sample rate in hertz."""
		return self._sample_rate

	def start(self) -> None:
		"""Open the microphone stream and begin a new in-memory recording."""
		with self._lock:
			if self._recording:
				raise AudioRecorderError("Recording is already in progress.")

			self._buffer.clear()
			self._recording = True
			try:
				stream = self._stream_factory(
					samplerate=self._sample_rate,
					channels=self._channels,
					dtype="int16",
					blocksize=self._block_size,
					callback=self._on_audio,
				)
				stream.start()
			except Exception as error:
				self._recording = False
				if "stream" in locals():
					stream.close()
				raise AudioRecorderError("Unable to start microphone capture.") from error

			self._stream = stream
			self._logger.debug("Started microphone capture at %s Hz.", self._sample_rate)

	def stop(self) -> bytes:
		"""Stop capture, close the stream, and return an immutable PCM snapshot."""
		with self._lock:
			if not self._recording:
				raise AudioRecorderError("No recording is in progress.")

			self._recording = False
			stream = self._stream
			self._stream = None
			audio = bytes(self._buffer)

		try:
			if stream is not None:
				stream.stop()
				stream.close()
		except Exception as error:
			raise AudioRecorderError("Unable to stop microphone capture cleanly.") from error

		self._logger.debug("Stopped microphone capture with %s bytes.", len(audio))
		return audio

	def get_audio(self) -> bytes:
		"""Return a consistent snapshot of the currently buffered PCM audio."""
		with self._lock:
			return bytes(self._buffer)

	def extract_and_clear_buffer(self) -> bytes:
		"""Extract buffered PCM audio and reset the buffer for the next chunk."""
		with self._lock:
			audio = bytes(self._buffer)
			self._buffer.clear()
			return audio

	def add_chunk_callback(self, callback: Callable[[bytes], None]) -> Callable[[], None]:
		"""Register a callback for live audio chunks and return an unregister function."""
		with self._lock:
			self._chunk_callbacks.append(callback)

		def remove_callback() -> None:
			with self._lock:
				if callback in self._chunk_callbacks:
					self._chunk_callbacks.remove(callback)

		return remove_callback

	def _on_audio(self, indata: Any, frames: int, _time: Any, status: Any) -> None:
		if status:
			self._logger.warning("Audio input status: %s", status)

		with self._lock:
			if not self._recording:
				return

			chunk = bytes(indata)
			remaining_capacity = self._max_buffer_size - len(self._buffer)
			if len(chunk) > remaining_capacity:
				self._buffer.extend(chunk[:remaining_capacity])
				self._logger.warning("Maximum recording duration reached; dropping later samples.")
			else:
				self._buffer.extend(chunk)
			callbacks = tuple(self._chunk_callbacks)

		for cb in callbacks:
			try:
				cb(chunk)
			except Exception:
				self._logger.exception("Audio chunk callback failed.")
