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


# Edit these values for normal day-to-day VoiceFlow behavior.
DEFAULT_SETTINGS = {
	"hotkey": "<ctrl>+<alt>+<space>",
	"hotkey_mode": "push_to_talk",
	"dictation_mode": "streaming",
	"pause_threshold_ms": 700,
	"sample_rate": 16_000,
	"channels": 1,
	"audio_block_duration_ms": 30,
	"max_recording_duration_seconds": 120,
	"vad_silence_duration_ms": 1_800,
	"vad_energy_threshold": 0.01,
	"stt_backend": "faster_whisper",
	"llm_backend": "ollama",
	"llm_timeout_seconds": 5.0,
	"whisper_model_size": "small.en",
	"whisper_device": "auto",
	"whisper_compute_type": "auto",
	"ollama_model": "qwen2.5:3b",
	"log_level": "DEBUG",
}


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
	dictation_mode: str = "streaming"
	pause_threshold_ms: int = 700
	whisper_model_size: str = "small.en"
	whisper_device: str = "auto"
	whisper_compute_type: str = "auto"
	ollama_model: str = "qwen2.5:3b"
	log_level: str = "DEBUG"

	def __post_init__(self) -> None:
		if self.hotkey_mode not in {"push_to_talk", "toggle"}:
			raise ValueError("hotkey_mode must be 'push_to_talk' or 'toggle'.")
		if self.dictation_mode not in {"streaming", "batch"}:
			raise ValueError("dictation_mode must be 'streaming' or 'batch'.")
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
		if self.pause_threshold_ms <= 0:
			raise ValueError("pause_threshold_ms must be positive.")


def load_settings() -> Settings:
	"""Load code defaults, with optional ``VOICEFLOW_*`` environment overrides."""
	return Settings(
		hotkey=os.getenv("VOICEFLOW_HOTKEY", DEFAULT_SETTINGS["hotkey"]),
		hotkey_mode=os.getenv("VOICEFLOW_HOTKEY_MODE", DEFAULT_SETTINGS["hotkey_mode"]).lower(),
		dictation_mode=os.getenv("VOICEFLOW_DICTATION_MODE", DEFAULT_SETTINGS["dictation_mode"]).lower(),
		pause_threshold_ms=_environment_int("VOICEFLOW_PAUSE_THRESHOLD_MS", DEFAULT_SETTINGS["pause_threshold_ms"]),
		sample_rate=_environment_int("VOICEFLOW_SAMPLE_RATE", DEFAULT_SETTINGS["sample_rate"]),
		channels=_environment_int("VOICEFLOW_CHANNELS", DEFAULT_SETTINGS["channels"]),
		audio_block_duration_ms=_environment_int("VOICEFLOW_AUDIO_BLOCK_MS", DEFAULT_SETTINGS["audio_block_duration_ms"]),
		max_recording_duration_seconds=_environment_int(
			"VOICEFLOW_MAX_RECORDING_SECONDS", DEFAULT_SETTINGS["max_recording_duration_seconds"]
		),
		vad_silence_duration_ms=_environment_int("VOICEFLOW_VAD_SILENCE_MS", DEFAULT_SETTINGS["vad_silence_duration_ms"]),
		vad_energy_threshold=_environment_float("VOICEFLOW_VAD_ENERGY_THRESHOLD", DEFAULT_SETTINGS["vad_energy_threshold"]),
		stt_backend=os.getenv("VOICEFLOW_STT_BACKEND", DEFAULT_SETTINGS["stt_backend"]),
		llm_backend=os.getenv("VOICEFLOW_LLM_BACKEND", DEFAULT_SETTINGS["llm_backend"]),
		llm_timeout_seconds=_environment_float("VOICEFLOW_LLM_TIMEOUT_SECONDS", DEFAULT_SETTINGS["llm_timeout_seconds"]),
		whisper_model_size=os.getenv("VOICEFLOW_WHISPER_MODEL", DEFAULT_SETTINGS["whisper_model_size"]),
		whisper_device=os.getenv("VOICEFLOW_WHISPER_DEVICE", DEFAULT_SETTINGS["whisper_device"]),
		whisper_compute_type=os.getenv("VOICEFLOW_WHISPER_COMPUTE_TYPE", DEFAULT_SETTINGS["whisper_compute_type"]),
		ollama_model=os.getenv("VOICEFLOW_OLLAMA_MODEL", DEFAULT_SETTINGS["ollama_model"]),
		log_level=os.getenv("VOICEFLOW_LOG_LEVEL", DEFAULT_SETTINGS["log_level"]).upper(),
	)


settings = load_settings()
