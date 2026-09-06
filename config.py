"""Application settings, model flags, and keybinding configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _environment_int(name: str, default: int) -> int:
	"""Return an integer environment setting with a clear error on invalid input."""
	value = os.getenv(name)
	if value is None:
		return default

	try:
		return int(value)
	except ValueError as error:
		raise ValueError(f"{name} must be an integer, got {value!r}.") from error


def _environment_float(name: str, default: float) -> float:
	"""Return a float environment setting with a clear error on invalid input."""
	value = os.getenv(name)
	if value is None:
		return default

	try:
		return float(value)
	except ValueError as error:
		raise ValueError(f"{name} must be a number, got {value!r}.") from error


@dataclass(frozen=True, slots=True)
class Settings:
	"""Runtime settings shared across the local dictation pipeline."""

	hotkey: str
	hotkey_mode: str
	sample_rate: int
	channels: int
	audio_block_duration_ms: int
	max_recording_duration_seconds: int
	vad_silence_duration_ms: int
	vad_energy_threshold: float
	stt_backend: str
	llm_backend: str
	llm_timeout_seconds: float
	whisper_model_size: str = "small.en"
	whisper_device: str = "auto"
	whisper_compute_type: str = "auto"
	ollama_model: str = "llama3.2:1b"
	log_level: str = "INFO"

	def __post_init__(self) -> None:
		if self.hotkey_mode not in {"push_to_talk", "toggle"}:
			raise ValueError("hotkey_mode must be 'push_to_talk' or 'toggle'.")
		if not self.hotkey.strip():
			raise ValueError("hotkey must not be empty.")
		if self.sample_rate <= 0 or self.channels <= 0:
			raise ValueError("sample_rate and channels must be positive.")
		if self.audio_block_duration_ms <= 0:
			raise ValueError("audio_block_duration_ms must be positive.")
		if self.max_recording_duration_seconds <= 0:
			raise ValueError("max_recording_duration_seconds must be positive.")
		if self.vad_silence_duration_ms < 0 or self.vad_energy_threshold < 0:
			raise ValueError("VAD settings cannot be negative.")
		if self.llm_timeout_seconds <= 0:
			raise ValueError("llm_timeout_seconds must be positive.")


def load_settings() -> Settings:
	"""Load settings from ``VOICEFLOW_*`` environment variables."""
	return Settings(
		hotkey=os.getenv("VOICEFLOW_HOTKEY", "<ctrl>+<alt>+<space>"),
		hotkey_mode=os.getenv("VOICEFLOW_HOTKEY_MODE", "push_to_talk").lower(),
		sample_rate=_environment_int("VOICEFLOW_SAMPLE_RATE", 16_000),
		channels=_environment_int("VOICEFLOW_CHANNELS", 1),
		audio_block_duration_ms=_environment_int("VOICEFLOW_AUDIO_BLOCK_MS", 30),
		max_recording_duration_seconds=_environment_int("VOICEFLOW_MAX_RECORDING_SECONDS", 120),
		vad_silence_duration_ms=_environment_int("VOICEFLOW_VAD_SILENCE_MS", 800),
		vad_energy_threshold=_environment_float("VOICEFLOW_VAD_ENERGY_THRESHOLD", 0.01),
		stt_backend=os.getenv("VOICEFLOW_STT_BACKEND", "mock"),
		llm_backend=os.getenv("VOICEFLOW_LLM_BACKEND", "mock"),
		llm_timeout_seconds=_environment_float("VOICEFLOW_LLM_TIMEOUT_SECONDS", 5.0),
		whisper_model_size=os.getenv("VOICEFLOW_WHISPER_MODEL", "small.en"),
		whisper_device=os.getenv("VOICEFLOW_WHISPER_DEVICE", "auto"),
		whisper_compute_type=os.getenv("VOICEFLOW_WHISPER_COMPUTE_TYPE", "auto"),
		ollama_model=os.getenv("VOICEFLOW_OLLAMA_MODEL", "llama3.2:1b"),
		log_level=os.getenv("VOICEFLOW_LOG_LEVEL", "INFO").upper(),
	)


settings = load_settings()
