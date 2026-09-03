"""Voice activity detection and endpointing."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass
import math
import threading

from config import Settings
from utils.logger import get_logger


@dataclass(frozen=True, slots=True)
class VADResult:
	"""The classification outcome for one PCM audio chunk."""

	energy: float
	is_speech: bool
	endpoint_reached: bool


EndpointCallback = Callable[[VADResult], None]


class VoiceActivityDetector:
	"""Detect speech and endpoint a recording after sustained silence."""

	_SAMPLE_WIDTH_BYTES = 2

	def __init__(self, settings: Settings) -> None:
		if settings.channels != 1:
			raise ValueError("VoiceActivityDetector supports mono PCM only.")

		self._sample_rate = settings.sample_rate
		self._energy_threshold = settings.vad_energy_threshold
		self._silence_sample_limit = math.ceil(
			settings.sample_rate * settings.vad_silence_duration_ms / 1_000
		)
		self._endpoint_callbacks: list[EndpointCallback] = []
		self._pending_byte = b""
		self._silence_samples = 0
		self._speech_detected = False
		self._endpoint_emitted = False
		self._lock = threading.RLock()
		self._logger = get_logger("vad")

	def add_endpoint_callback(self, callback: EndpointCallback) -> Callable[[], None]:
		"""Register a callback and return a function that unregisters it."""
		with self._lock:
			self._endpoint_callbacks.append(callback)

		def remove_callback() -> None:
			with self._lock:
				if callback in self._endpoint_callbacks:
					self._endpoint_callbacks.remove(callback)

		return remove_callback

	def reset(self) -> None:
		"""Clear speech and silence history before a new recording begins."""
		with self._lock:
			self._pending_byte = b""
			self._silence_samples = 0
			self._speech_detected = False
			self._endpoint_emitted = False

	def process(self, pcm_audio: bytes) -> VADResult:
		"""Analyze an ``int16`` PCM chunk and emit an endpoint once when appropriate."""
		with self._lock:
			pcm_audio = self._pending_byte + pcm_audio
			self._pending_byte = pcm_audio[-1:] if len(pcm_audio) % self._SAMPLE_WIDTH_BYTES else b""
			aligned_audio = pcm_audio[:-1] if self._pending_byte else pcm_audio
			energy, sample_count = self._calculate_energy(aligned_audio)
			is_speech = sample_count > 0 and energy >= self._energy_threshold

			if is_speech:
				self._speech_detected = True
				self._silence_samples = 0
			elif self._speech_detected:
				self._silence_samples += sample_count

			endpoint_reached = (
				self._speech_detected
				and not self._endpoint_emitted
				and self._silence_samples >= self._silence_sample_limit
			)
			if endpoint_reached:
				self._endpoint_emitted = True
				callbacks = tuple(self._endpoint_callbacks)
			else:
				callbacks = ()

			result = VADResult(energy, is_speech, endpoint_reached)
		for callback in callbacks:
			try:
				callback(result)
			except Exception:
				self._logger.exception("VAD endpoint callback failed.")
		return result

	@staticmethod
	def _calculate_energy(pcm_audio: bytes) -> tuple[float, int]:
		if not pcm_audio:
			return 0.0, 0

		samples = array("h")
		samples.frombytes(pcm_audio)
		mean_square = sum(sample * sample for sample in samples) / len(samples)
		return math.sqrt(mean_square) / 32_768, len(samples)
