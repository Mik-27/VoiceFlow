"""Launch supported desktop applications."""

from __future__ import annotations

import os
import subprocess

from utils.logger import get_logger


class AppLauncher:
	"""Launch supported desktop applications on Windows."""

	APP_MAPPINGS = {
		"WORD": "winword.exe",
		"EXCEL": "excel.exe",
		"POWERPOINT": "powerpnt.exe",
		"CHROME": "chrome.exe",
		"NOTEPAD": "notepad.exe",
		"CALCULATOR": "calc.exe",
		"VS_CODE": "code",
		"TERMINAL": "cmd.exe",
	}

	def __init__(self) -> None:
		self._logger = get_logger("app_launcher")

	def open_app(self, app_name: str) -> bool:
		"""Launch a supported application and return whether its process was started."""
		if not isinstance(app_name, str):
			raise TypeError("app_name must be a string.")

		executable = self.APP_MAPPINGS.get(app_name.upper())
		if executable is None:
			self._logger.warning("Unsupported application command: %r", app_name)
			return False

		try:
			startfile = getattr(os, "startfile", None)
			if startfile is not None:
				startfile(executable)
			else:
				subprocess.Popen([executable])
			return True
		except FileNotFoundError:
			try:
				subprocess.Popen([executable])
				return True
			except OSError:
				self._logger.exception("Unable to launch %s.", app_name)
				return False
		except OSError:
			self._logger.exception("Unable to launch %s.", app_name)
			return False