"""Application entry point and orchestration layer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import signal
import sys
import threading
import time

from config import Settings, load_settings, settings as default_settings
from core.audio_recorder import AudioRecorder
from core.injection import TextInjector
from core.llm_cleaner import BaseLLMCleaner, MockLLMCleaner, OllamaCleaner
from core.state_machine import DictationStateMachine, PipelineState
from core.stt_engine import BaseSTTEngine, FasterWhisperEngine, MockSTTEngine
from core.vad import VADResult, VoiceActivityDetector
from utils.hotkey_listener import HotkeyEvent, HotkeyListener
from utils.logger import configure_logging, get_logger


def create_stt_engine(settings: Settings) -> BaseSTTEngine:
	"""Instantiate the configured STT engine backend."""
	backend = settings.stt_backend.lower().strip()
	if backend in {"faster_whisper", "whisper"}:
		return FasterWhisperEngine(
			model_size=settings.whisper_model_size,
			device=settings.whisper_device,
			compute_type=settings.whisper_compute_type,
		)
	if backend == "mock":
		return MockSTTEngine()
	raise ValueError(f"Unknown STT backend: {settings.stt_backend!r}.")


def create_llm_cleaner(settings: Settings) -> BaseLLMCleaner:
	"""Instantiate the configured LLM cleaner backend."""
	backend = settings.llm_backend.lower().strip()
	if backend == "ollama":
		return OllamaCleaner(model_name=settings.ollama_model)
	if backend == "mock":
		return MockLLMCleaner()
	raise ValueError(f"Unknown LLM backend: {settings.llm_backend!r}.")


class VoiceFlowApp:
	"""Orchestrate the end-to-end local voice dictation pipeline."""

	def __init__(
		self,
		settings: Settings | None = None,
		recorder: AudioRecorder | None = None,
		vad: VoiceActivityDetector | None = None,
		stt_engine: BaseSTTEngine | None = None,
		llm_cleaner: BaseLLMCleaner | None = None,
		injector: TextInjector | None = None,
		state_machine: DictationStateMachine | None = None,
		hotkey_listener: HotkeyListener | None = None,
	) -> None:
		self.settings = settings or default_settings
		self.logger = get_logger("app")

		self.state_machine = state_machine or DictationStateMachine()
		self.recorder = recorder or AudioRecorder(self.settings)
		self.vad = vad or VoiceActivityDetector(self.settings)
		self.stt_engine = stt_engine or create_stt_engine(self.settings)
		self.llm_cleaner = llm_cleaner or create_llm_cleaner(self.settings)
		self.injector = injector or TextInjector()

		self.hotkey_listener = hotkey_listener or HotkeyListener(
			hotkey=self.settings.hotkey,
			mode=self.settings.hotkey_mode,
		)

		self._pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voiceflow-pipeline")
		self._streaming_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voiceflow-stream")
		self._lock = threading.RLock()
		self._streaming_lock = threading.Lock()
		self._has_typed_any_phrase = False
		self._running = False
		self._remove_callbacks: list[tuple[str, object]] = []

		self._setup_wiring()

	def _setup_wiring(self) -> None:
		"""Connect audio, VAD, hotkey, and state machine event handlers."""
		self.hotkey_listener.add_callback(self._on_hotkey_event)
		self.recorder.add_chunk_callback(self._on_audio_chunk)
		self.vad.add_pause_callback(self._on_vad_pause)
		self.vad.add_endpoint_callback(self._on_vad_endpoint)

	def start(self) -> None:
		"""Start the application and global hotkey listener."""
		with self._lock:
			if self._running:
				return
			self._running = True
			self.hotkey_listener.start()
			self.logger.info(
				"VoiceFlow is running. Hotkey: %s (%s mode).",
				self.settings.hotkey,
				self.settings.hotkey_mode,
			)

	def stop(self) -> None:
		"""Stop the application cleanly."""
		with self._lock:
			if not self._running:
				return
			self._running = False

		self.hotkey_listener.stop()
		if self.recorder.is_recording:
			try:
				self.recorder.stop()
			except Exception:
				pass
		self.state_machine.cancel()
		self._streaming_executor.shutdown(wait=False, cancel_futures=True)
		self._pipeline_executor.shutdown(wait=False, cancel_futures=True)
		self.logger.info("VoiceFlow stopped.")

	def _on_hotkey_event(self, event: HotkeyEvent) -> None:
		"""Handle global hotkey press and release events."""
		if event is HotkeyEvent.ON_PRESS:
			self._begin_recording()
		elif event is HotkeyEvent.ON_RELEASE:
			self._stop_recording_and_process()

	def _on_audio_chunk(self, chunk: bytes) -> None:
		"""Feed live PCM audio chunks into VAD."""
		if self.state_machine.state is PipelineState.RECORDING:
			self.vad.process(chunk)

	def _on_vad_pause(self, result: VADResult) -> None:
		"""Trigger incremental phrase processing on natural speech pauses in streaming mode."""
		if self.settings.dictation_mode == "streaming" and self.state_machine.state is PipelineState.RECORDING:
			self._streaming_executor.submit(self._process_phrase_chunk)

	def _process_phrase_chunk(self) -> None:
		"""Process an individual spoken phrase chunk with STT and LLM cleanup in real time."""
		if self.state_machine.state is not PipelineState.RECORDING:
			return

		with self._streaming_lock:
			if self.state_machine.state is not PipelineState.RECORDING:
				return

			# Extract only the audio for this phrase and reset recorder buffer
			phrase_audio = self.recorder.extract_and_clear_buffer()
			self.vad.reset_phrase()

			# Need at least 0.3s of audio to be a meaningful phrase
			min_samples = int(self.settings.sample_rate * 2 * 0.3)
			if len(phrase_audio) < min_samples:
				return

			try:
				raw_phrase = self.stt_engine.transcribe(phrase_audio).strip()
			except Exception:
				self.logger.debug("Phrase transcription error (ignored).")
				return

			if not raw_phrase:
				return

			self.logger.info("Live phrase raw STT: %r", raw_phrase)

			# Fast LLM cleaning on the single phrase
			cleaned_phrase = self._clean_phrase_with_fallback(raw_phrase)
			if cleaned_phrase:
				prefix = " " if self._has_typed_any_phrase else ""
				to_type = prefix + cleaned_phrase
				self.logger.info("Real-time typing phrase: %r", to_type)
				self.injector.type_text(to_type)
				self._has_typed_any_phrase = True

	def _clean_phrase_with_fallback(self, raw_text: str) -> str:
		"""Fast LLM cleanup of an individual phrase with graceful fallback."""
		def _collect_tokens() -> str:
			chunks = list(self.llm_cleaner.clean_stream(raw_text))
			return "".join(chunks).strip()

		cleanup_executor = ThreadPoolExecutor(max_workers=1)
		try:
			future = cleanup_executor.submit(_collect_tokens)
			cleaned = future.result(timeout=self.settings.llm_timeout_seconds)
			if cleaned:
				return cleaned
			return raw_text
		except Exception:
			return raw_text
		finally:
			cleanup_executor.shutdown(wait=False, cancel_futures=True)

	def _on_vad_endpoint(self, result: VADResult) -> None:
		"""Automatically trigger end of recording when VAD detects silence endpoint."""
		if result.endpoint_reached:
			self.logger.info("VAD silence endpoint reached; stopping recording.")
			self._stop_recording_and_process()

	def _begin_recording(self) -> None:
		"""Start microphone capture if in IDLE state."""
		with self._lock:
			if not self.state_machine.begin_recording():
				return

			with self._streaming_lock:
				self._has_typed_any_phrase = False

			self.vad.reset()
			try:
				self.recorder.start()
				self.logger.info("Recording started (%s mode)...", self.settings.dictation_mode)
			except Exception:
				self.logger.exception("Failed to start audio recording.")
				self.state_machine.cancel()

	def _stop_recording_and_process(self) -> None:
		"""Stop microphone capture and dispatch the processing pipeline."""
		with self._lock:
			if self.state_machine.state is not PipelineState.RECORDING:
				return

			try:
				audio_bytes = self.recorder.stop()
			except Exception:
				self.logger.exception("Failed to stop audio recording cleanly.")
				audio_bytes = b""

			if not self.state_machine.finish_recording():
				return

			self.logger.info("Recording finished (%d bytes). Processing final pipeline...", len(audio_bytes))
			self._pipeline_executor.submit(self._run_pipeline, audio_bytes)

	def _run_pipeline(self, audio_bytes: bytes) -> None:
		"""Process any remaining audio and finalize dictation."""
		if not audio_bytes:
			if not self._has_typed_any_phrase:
				self.logger.warning("No audio data captured.")
			self.state_machine.cancel()
			return

		# 1. Transcribe (Pass 1)
		try:
			self.logger.debug("Transcribing %d bytes of remaining PCM audio...", len(audio_bytes))
			raw_text = self.stt_engine.transcribe(audio_bytes).strip()
		except Exception:
			self.logger.exception("STT transcription failed.")
			self.state_machine.cancel()
			return

		if not raw_text:
			self.state_machine.cancel()
			return

		self.logger.info("Remaining audio raw transcript: %r", raw_text)

		# 2. In streaming mode, clean and type the remaining phrase directly
		if self.settings.dictation_mode == "streaming":
			with self._streaming_lock:
				cleaned_phrase = self._clean_phrase_with_fallback(raw_text)
				if cleaned_phrase:
					prefix = " " if self._has_typed_any_phrase else ""
					to_type = prefix + cleaned_phrase
					self.logger.info("Final phrase typed: %r", to_type)
					self.injector.type_text(to_type)
					self._has_typed_any_phrase = True

			self.state_machine.finish_transcription(needs_cleaning=False)
			self.state_machine.finish_injection()
			self.logger.info("Streaming dictation complete. Ready for next dictation.")
			return

		# Batch Mode: 2. LLM Cleanup (Pass 2) with graceful fallback
		cleaned_text = self._clean_text_with_fallback(raw_text)

		if not cleaned_text:
			self.logger.info("Empty cleaned text; nothing to inject.")
			self.state_machine.cancel()
			return

		self.logger.info("Final text to inject: %r", cleaned_text)

		# Batch Mode: 3. Clipboard Text Injection
		try:
			self.state_machine.finish_cleaning()
			self.logger.debug("Injecting text into active context...")
			self.injector.inject(cleaned_text)
		except Exception:
			self.logger.exception("Text injection failed.")
		finally:
			self.state_machine.finish_injection()
			self.logger.info("Pipeline complete. Ready for next dictation.")

		if not cleaned_text:
			self.logger.info("Empty cleaned text; nothing to inject.")
			self.state_machine.cancel()
			return

		self.logger.info("Final text to inject: %r", cleaned_text)

		# Batch Mode: 3. Clipboard Text Injection
		try:
			self.state_machine.finish_cleaning()
			self.logger.debug("Injecting text into active context...")
			self.injector.inject(cleaned_text)
		except Exception:
			self.logger.exception("Text injection failed.")
		finally:
			self.state_machine.finish_injection()
			self.logger.info("Pipeline complete. Ready for next dictation.")

	def _clean_text_with_fallback(self, raw_text: str) -> str:
		"""Clean transcript using LLM with timeout and graceful fallback to raw text."""
		if not self.state_machine.finish_transcription(needs_cleaning=True):
			return raw_text

		def _collect_tokens() -> str:
			chunks = list(self.llm_cleaner.clean_stream(raw_text))
			return "".join(chunks).strip()

		cleanup_executor = ThreadPoolExecutor(max_workers=1)
		try:
			future = cleanup_executor.submit(_collect_tokens)
			cleaned = future.result(timeout=self.settings.llm_timeout_seconds)
			if cleaned:
				return cleaned
			self.logger.warning("LLM cleaner returned empty response; falling back to raw text.")
			return raw_text
		except FutureTimeoutError:
			self.logger.warning(
				"LLM cleaner timed out after %.1fs; falling back to raw transcript.",
				self.settings.llm_timeout_seconds,
			)
			return raw_text
		except Exception as error:
			self.logger.warning(
				"LLM cleaner failed (%s); falling back to raw transcript.",
				error,
			)
			return raw_text
		finally:
			cleanup_executor.shutdown(wait=False, cancel_futures=True)


def main() -> None:
	"""Application entry point."""
	app_settings = load_settings()
	configure_logging(app_settings.log_level)
	logger = get_logger("main")

	logger.info("Initializing VoiceFlow local voice dictation system...")
	app = VoiceFlowApp(settings=app_settings)
	app.start()

	stop_event = threading.Event()

	def _signal_handler(signum: int, frame: object) -> None:
		logger.info("Termination signal received. Shutting down...")
		stop_event.set()

	signal.signal(signal.SIGINT, _signal_handler)
	signal.signal(signal.SIGTERM, _signal_handler)

	logger.info("Press Ctrl+C in terminal to exit.")
	try:
		while not stop_event.is_set():
			stop_event.wait(timeout=0.5)
	except KeyboardInterrupt:
		logger.info("Keyboard interrupt received.")
	finally:
		app.stop()


if __name__ == "__main__":
	main()
