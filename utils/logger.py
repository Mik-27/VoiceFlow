"""Application logging configuration."""

from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
	"""Configure VoiceFlow's console logger without duplicating handlers."""
	numeric_level = logging.getLevelName(level.upper())
	if not isinstance(numeric_level, int):
		raise ValueError(f"Unknown logging level: {level!r}.")

	logger = logging.getLogger("voiceflow")
	logger.setLevel(numeric_level)
	logger.propagate = False

	if logger.handlers:
		for handler in logger.handlers:
			handler.setLevel(numeric_level)
		return

	handler = logging.StreamHandler(sys.stderr)
	handler.setLevel(numeric_level)
	handler.setFormatter(logging.Formatter(LOG_FORMAT))
	logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
	"""Return a logger scoped beneath the VoiceFlow application namespace."""
	if not name:
		raise ValueError("Logger name must not be empty.")
	return logging.getLogger(f"voiceflow.{name}")
