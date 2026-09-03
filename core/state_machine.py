"""Thread-safe application pipeline state transitions."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import threading
from typing import Final

from utils.hotkey_listener import HotkeyEvent
from utils.logger import get_logger


class PipelineState(str, Enum):
	"""The stages of a single dictation request."""

	IDLE = "idle"
	RECORDING = "recording"
	TRANSCRIBING = "transcribing"
	CLEANING = "cleaning"
	INJECTING = "injecting"


class InvalidStateTransition(RuntimeError):
	"""Raised when a pipeline transition is not valid for the current state."""


StateChangeCallback = Callable[[PipelineState, PipelineState], None]


class DictationStateMachine:
	"""Coordinate dictation lifecycle transitions across hotkey and worker threads."""

	_ALLOWED_TRANSITIONS: Final[dict[PipelineState, frozenset[PipelineState]]] = {
		PipelineState.IDLE: frozenset({PipelineState.RECORDING}),
		PipelineState.RECORDING: frozenset({PipelineState.TRANSCRIBING, PipelineState.IDLE}),
		PipelineState.TRANSCRIBING: frozenset({PipelineState.CLEANING, PipelineState.INJECTING, PipelineState.IDLE}),
		PipelineState.CLEANING: frozenset({PipelineState.INJECTING, PipelineState.IDLE}),
		PipelineState.INJECTING: frozenset({PipelineState.IDLE}),
	}

	def __init__(self) -> None:
		self._state = PipelineState.IDLE
		self._callbacks: list[StateChangeCallback] = []
		self._condition = threading.Condition(threading.RLock())
		self._logger = get_logger("state_machine")

	@property
	def state(self) -> PipelineState:
		"""Return a consistent snapshot of the current pipeline state."""
		with self._condition:
			return self._state

	def add_callback(self, callback: StateChangeCallback) -> Callable[[], None]:
		"""Register a state-change callback and return an unregister function."""
		with self._condition:
			self._callbacks.append(callback)

		def remove_callback() -> None:
			with self._condition:
				if callback in self._callbacks:
					self._callbacks.remove(callback)

		return remove_callback

	def handle_hotkey_event(self, event: HotkeyEvent) -> bool:
		"""Apply a hotkey event, returning ``False`` when it is not actionable."""
		if event is HotkeyEvent.ON_PRESS:
			return self.begin_recording()
		if event is HotkeyEvent.ON_RELEASE:
			return self.finish_recording()
		raise ValueError(f"Unsupported hotkey event: {event!r}.")

	def begin_recording(self) -> bool:
		"""Move from idle to recording when no dictation is already in progress."""
		return self._try_transition(PipelineState.RECORDING)

	def finish_recording(self) -> bool:
		"""Move from recording to transcription after audio capture ends."""
		return self._try_transition(PipelineState.TRANSCRIBING)

	def finish_transcription(self, needs_cleaning: bool = True) -> bool:
		"""Advance a transcript to cleanup or directly to text injection."""
		target = PipelineState.CLEANING if needs_cleaning else PipelineState.INJECTING
		return self._try_transition(target)

	def finish_cleaning(self) -> bool:
		"""Move cleaned text to the injection stage."""
		return self._try_transition(PipelineState.INJECTING)

	def finish_injection(self) -> bool:
		"""Return to idle after text has been injected."""
		return self._try_transition(PipelineState.IDLE)

	def cancel(self) -> bool:
		"""Abort an active dictation request and return to idle."""
		with self._condition:
			if self._state is PipelineState.IDLE:
				return False
			previous, callbacks = self._transition(PipelineState.IDLE)
		self._notify_callbacks(previous, PipelineState.IDLE, callbacks)
		return True

	def wait_for_state(self, state: PipelineState, timeout: float | None = None) -> bool:
		"""Wait until the machine reaches ``state`` or the timeout expires."""
		with self._condition:
			return self._condition.wait_for(lambda: self._state is state, timeout)

	def _try_transition(self, target: PipelineState) -> bool:
		with self._condition:
			if target not in self._ALLOWED_TRANSITIONS[self._state]:
				return False
			previous, callbacks = self._transition(target)
		self._notify_callbacks(previous, target, callbacks)
		return True

	def _transition(self, target: PipelineState) -> tuple[PipelineState, tuple[StateChangeCallback, ...]]:
		previous = self._state
		if target not in self._ALLOWED_TRANSITIONS[previous]:
			raise InvalidStateTransition(f"Cannot transition from {previous.value} to {target.value}.")

		self._state = target
		callbacks = tuple(self._callbacks)
		self._condition.notify_all()
		self._logger.debug("Pipeline state changed from %s to %s.", previous.value, target.value)
		return previous, callbacks

	def _notify_callbacks(
		self,
		previous: PipelineState,
		target: PipelineState,
		callbacks: tuple[StateChangeCallback, ...],
	) -> None:
		for callback in callbacks:
			try:
				callback(previous, target)
			except Exception:
				self._logger.exception("State-change callback failed.")
