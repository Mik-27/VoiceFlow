"""Non-blocking global push-to-talk and toggle hotkey listener."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import queue
import threading
from typing import Any

from pynput import keyboard

from utils.logger import get_logger


class HotkeyEvent(str, Enum):
	"""Events emitted when the configured dictation hotkey changes state."""

	ON_PRESS = "on_press"
	ON_RELEASE = "on_release"


HotkeyCallback = Callable[[HotkeyEvent], None]
ListenerFactory = Callable[..., Any]


class HotkeyListener:
	"""Listen for a global hotkey without blocking the OS keyboard hook thread."""

	def __init__(
		self,
		hotkey: str,
		mode: str = "push_to_talk",
		listener_factory: ListenerFactory | None = None,
	) -> None:
		if mode not in {"push_to_talk", "toggle"}:
			raise ValueError("mode must be 'push_to_talk' or 'toggle'.")

		try:
			self._hotkey_keys = frozenset(keyboard.HotKey.parse(hotkey))
		except (KeyError, ValueError) as error:
			raise ValueError(f"Invalid hotkey: {hotkey!r}.") from error

		if not self._hotkey_keys:
			raise ValueError("hotkey must contain at least one key.")

		self._hotkey = hotkey
		self._mode = mode
		self._listener_factory = listener_factory or keyboard.Listener
		self._callbacks: list[HotkeyCallback] = []
		self._pressed_keys: set[Any] = set()
		self._event_queue: queue.Queue[HotkeyEvent | None] = queue.Queue()
		self._listener: Any | None = None
		self._dispatcher: threading.Thread | None = None
		self._active = False
		self._running = False
		self._lock = threading.RLock()
		self._logger = get_logger("hotkey")

	def add_callback(self, callback: HotkeyCallback) -> Callable[[], None]:
		"""Register a callback and return a function that unregisters it."""
		with self._lock:
			self._callbacks.append(callback)

		def remove_callback() -> None:
			with self._lock:
				if callback in self._callbacks:
					self._callbacks.remove(callback)

		return remove_callback

	def start(self) -> None:
		"""Start keyboard monitoring in the background."""
		with self._lock:
			if self._running:
				return

			self._running = True
			self._dispatcher = threading.Thread(
				target=self._dispatch_events,
				name="voiceflow-hotkey-events",
				daemon=True,
			)
			self._dispatcher.start()
			self._listener = self._listener_factory(
				on_press=self._on_press,
				on_release=self._on_release,
			)
			self._listener.start()
			self._logger.info("Started %s hotkey listener for %s.", self._mode, self._hotkey)

	def stop(self) -> None:
		"""Stop global keyboard monitoring and its event dispatcher."""
		with self._lock:
			if not self._running:
				return

			self._running = False
			listener = self._listener
			dispatcher = self._dispatcher
			self._listener = None
			self._dispatcher = None
			self._pressed_keys.clear()
			self._active = False

		if listener is not None:
			listener.stop()
		self._event_queue.put(None)
		if dispatcher is not None and dispatcher is not threading.current_thread():
			dispatcher.join(timeout=1.0)
		self._logger.info("Stopped hotkey listener.")

	def _on_press(self, key: Any) -> None:
		canonical_key = self._canonical_key(key)
		with self._lock:
			hotkey_was_pressed = self._hotkey_keys.issubset(self._pressed_keys)
			self._pressed_keys.add(canonical_key)
			if self._hotkey_keys.issubset(self._pressed_keys) and not hotkey_was_pressed:
				if self._mode == "toggle":
					event = HotkeyEvent.ON_RELEASE if self._active else HotkeyEvent.ON_PRESS
					self._active = not self._active
					self._event_queue.put(event)
				elif not self._active:
					self._active = True
					self._event_queue.put(HotkeyEvent.ON_PRESS)

	def _on_release(self, key: Any) -> None:
		canonical_key = self._canonical_key(key)
		with self._lock:
			self._pressed_keys.discard(canonical_key)
			if (
				self._mode == "push_to_talk"
				and self._active
				and not self._hotkey_keys.issubset(self._pressed_keys)
			):
				self._active = False
				self._event_queue.put(HotkeyEvent.ON_RELEASE)

	@staticmethod
	def _canonical_key(key: Any) -> Any:
		if isinstance(key, keyboard.KeyCode) and key.char is not None:
			return keyboard.KeyCode.from_char(key.char.lower())

		key_aliases = {
			keyboard.Key.alt_l: keyboard.Key.alt,
			keyboard.Key.alt_r: keyboard.Key.alt,
			keyboard.Key.ctrl_l: keyboard.Key.ctrl,
			keyboard.Key.ctrl_r: keyboard.Key.ctrl,
			keyboard.Key.shift_l: keyboard.Key.shift,
			keyboard.Key.shift_r: keyboard.Key.shift,
		}
		return key_aliases.get(key, key)

	def _dispatch_events(self) -> None:
		while True:
			event = self._event_queue.get()
			if event is None:
				return

			with self._lock:
				callbacks = tuple(self._callbacks)
			for callback in callbacks:
				try:
					callback(event)
				except Exception:
					self._logger.exception("Hotkey callback failed for event %s.", event.value)
