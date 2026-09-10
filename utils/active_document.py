"""Detect applications where formatting commands can be safely executed."""

from __future__ import annotations

from collections.abc import Iterable
import ctypes
from ctypes import wintypes
import os
from typing import Protocol


class CommandTarget(Protocol):
	"""An application target that can receive formatting commands."""

	def is_active(self) -> bool:
		"""Return whether this target has the active document cursor."""


class ActiveDocumentTarget:
	"""Determine whether any supported document target is active."""

	def __init__(self, targets: Iterable[CommandTarget] | None = None) -> None:
		self._targets = tuple(targets or (WordDocumentTarget(),))

	def is_active(self) -> bool:
		"""Return whether a supported application owns the active document cursor."""
		return any(target.is_active() for target in self._targets)


class WordDocumentTarget:
	"""Detect the Microsoft Word document window as the active target."""

	def is_active(self) -> bool:
		"""Return whether the foreground window belongs to Microsoft Word."""
		if os.name != "nt":
			return False

		user32 = ctypes.WinDLL("user32", use_last_error=True)
		kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
		user32.GetForegroundWindow.restype = wintypes.HWND
		user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
		user32.GetClassNameW.restype = ctypes.c_int
		user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
		user32.GetWindowThreadProcessId.restype = wintypes.DWORD
		kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
		kernel32.OpenProcess.restype = wintypes.HANDLE
		kernel32.QueryFullProcessImageNameW.argtypes = (
			wintypes.HANDLE,
			wintypes.DWORD,
			wintypes.LPWSTR,
			ctypes.POINTER(wintypes.DWORD),
		)
		kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
		kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
		kernel32.CloseHandle.restype = wintypes.BOOL

		foreground_window = user32.GetForegroundWindow()
		if not foreground_window:
			return False

		class_name = ctypes.create_unicode_buffer(256)
		if not user32.GetClassNameW(foreground_window, class_name, len(class_name)):
			return False
		if class_name.value != "OpusApp":
			return False

		process_id = wintypes.DWORD()
		user32.GetWindowThreadProcessId(foreground_window, ctypes.byref(process_id))
		process = kernel32.OpenProcess(0x1000, False, process_id.value)
		if not process:
			return False

		try:
			executable_path = ctypes.create_unicode_buffer(260)
			path_size = wintypes.DWORD(len(executable_path))
			success = kernel32.QueryFullProcessImageNameW(
				process,
				0,
				executable_path,
				ctypes.byref(path_size),
			)
			return bool(success) and os.path.basename(executable_path.value).upper() == "WINWORD.EXE"
		finally:
			kernel32.CloseHandle(process)