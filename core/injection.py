"""Operating-system text injection and clipboard management."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import TYPE_CHECKING, Protocol

import pyperclip
from pynput import keyboard

from core.app_launcher import AppLauncher
from core.word_formatter import WordFormatter
from utils.logger import get_logger

if TYPE_CHECKING:
	from core.computer_use import CUAgent


class Clipboard(Protocol):
	"""Text clipboard operations required by the injector."""

	def paste(self) -> str:
		"""Return the current clipboard text."""

	def copy(self, text: str) -> None:
		"""Replace the clipboard text."""


class PyperclipClipboard:
	"""Clipboard adapter backed by pyperclip."""

	def paste(self) -> str:
		return pyperclip.paste()

	def copy(self, text: str) -> None:
		pyperclip.copy(text)


class TextInjector:
	"""Inject text through native typing or clipboard paste into the focused application."""

	def __init__(
		self,
		clipboard: Clipboard | None = None,
		key_controller: keyboard.Controller | None = None,
		formatter: WordFormatter | None = None,
		app_launcher: AppLauncher | None = None,
		computer_use_agent: CUAgent | None = None,
		restore_delay_seconds: float = 0.05,
		sleep: Callable[[float], None] = time.sleep,
	) -> None:
		if restore_delay_seconds < 0:
			raise ValueError("restore_delay_seconds cannot be negative.")

		self._clipboard = clipboard or PyperclipClipboard()
		self._key_controller = key_controller or keyboard.Controller()
		self._formatter = formatter or WordFormatter()
		self._app_launcher = app_launcher or AppLauncher()
		self._computer_use_agent = computer_use_agent
		self._restore_delay_seconds = restore_delay_seconds
		self._sleep = sleep
		self._lock = threading.Lock()
		self._logger = get_logger("injection")

	def handle_payload(self, payload: dict[str, str] | None) -> None:
		"""Route an LLM payload to formatting, app launch, computer use, or native text injection."""
		if not payload:
			return

		payload_type = payload.get("type")
		self._logger.debug("Handling payload of type: %s", payload_type)
		if payload_type == "COMMAND":
			self._formatter.execute_formatting(payload.get("action", ""))
		elif payload_type == "APP_COMMAND":
			self._app_launcher.open_app(payload.get("app", ""))
		elif payload_type == "ACTION_AGENT":
			goal = payload.get("goal") or payload.get("text", "")
			if self._computer_use_agent is None:
				from core.computer_use import CUAgent
				self._computer_use_agent = CUAgent()
			self._computer_use_agent.execute_action(goal)
		elif payload_type == "DICTATION":
			self.inject_text(payload.get("text", ""))

	def inject(self, text: str) -> None:
		"""Paste ``text`` into the focused window and restore the prior text clipboard."""
		if not isinstance(text, str):
			raise TypeError("text must be a string.")
		if not text:
			return

		with self._lock:
			previous_clipboard: str | None = None
			try:
				previous_clipboard = self._clipboard.paste()
				self._clipboard.copy(text)
				self._paste()
				self._sleep(self._restore_delay_seconds)
			finally:
				if previous_clipboard is not None:
					try:
						self._clipboard.copy(previous_clipboard)
					except Exception:
						self._logger.exception("Unable to restore the previous clipboard text.")

	def inject_text(self, text: str) -> None:
		"""Type ``text`` as native keyboard events at the active cursor position."""
		if not isinstance(text, str):
			raise TypeError("text must be a string.")
		if not text:
			return

		with self._lock:
			try:
				self._key_controller.type(text)
			except Exception:
				self._logger.exception("Failed to inject text through native keyboard events.")

	def type_text(self, text: str) -> None:
		"""Compatibility alias for native character-based text injection."""
		self.inject_text(text)

	def _paste(self) -> None:
		self._key_controller.press(keyboard.Key.ctrl)
		try:
			self._key_controller.press("v")
			self._key_controller.release("v")
		finally:
			self._key_controller.release(keyboard.Key.ctrl)
