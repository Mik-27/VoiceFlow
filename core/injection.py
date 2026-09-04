"""Operating-system text injection and clipboard management."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Protocol

import pyperclip
from pynput import keyboard

from utils.logger import get_logger


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
	"""Paste text into the currently focused application and restore its clipboard."""

	def __init__(
		self,
		clipboard: Clipboard | None = None,
		key_controller: keyboard.Controller | None = None,
		restore_delay_seconds: float = 0.05,
		sleep: Callable[[float], None] = time.sleep,
	) -> None:
		if restore_delay_seconds < 0:
			raise ValueError("restore_delay_seconds cannot be negative.")

		self._clipboard = clipboard or PyperclipClipboard()
		self._key_controller = key_controller or keyboard.Controller()
		self._restore_delay_seconds = restore_delay_seconds
		self._sleep = sleep
		self._lock = threading.Lock()
		self._logger = get_logger("injection")

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

	def _paste(self) -> None:
		self._key_controller.press(keyboard.Key.ctrl)
		try:
			self._key_controller.press("v")
			self._key_controller.release("v")
		finally:
			self._key_controller.release(keyboard.Key.ctrl)
