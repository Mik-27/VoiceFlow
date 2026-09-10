"""Transcript-cleaning language model interfaces and prompts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from enum import Enum
import json
import re
from typing import Literal

import ollama
from pydantic import BaseModel, model_validator

from config.prompts import WORD_FORMATTING_SYSTEM_PROMPT


class FormattingAction(str, Enum):
	"""Microsoft Word actions supported by the formatter."""

	TOGGLE_BULLETS = "TOGGLE_BULLETS"
	FONT_INCREASE = "FONT_INCREASE"
	FONT_DECREASE = "FONT_DECREASE"
	BOLD = "BOLD"
	ITALIC = "ITALIC"
	UNDERLINE = "UNDERLINE"
	ALIGN_LEFT = "ALIGN_LEFT"
	ALIGN_CENTER = "ALIGN_CENTER"
	ALIGN_RIGHT = "ALIGN_RIGHT"


class TranscriptPayload(BaseModel):
	"""Validated LLM classification result for a transcribed phrase."""

	type: Literal["COMMAND", "DICTATION"]
	action: FormattingAction | None = None
	text: str | None = None

	@model_validator(mode="after")
	def validate_intent_fields(self) -> TranscriptPayload:
		if self.type == "COMMAND" and self.action is None:
			raise ValueError("COMMAND payloads require an action.")
		if self.type == "DICTATION" and self.text is None:
			raise ValueError("DICTATION payloads require text.")
		return self


class BaseLLMCleaner(ABC):
	"""Classify a transcript as dictation text or a formatting command."""

	@abstractmethod
	def process_transcript(self, raw_text: str) -> dict[str, str]:
		"""Return a validated command or dictation payload."""

	def clean_stream(self, raw_text: str) -> Generator[str, None, None]:
		"""Yield dictation text for callers using the original cleaner API."""
		payload = self.process_transcript(raw_text)
		if payload["type"] == "DICTATION" and payload["text"]:
			yield payload["text"]

	def clean_text(self, raw_text: str) -> Generator[str, None, None]:
		"""Compatibility alias for callers using the original cleaner API."""
		yield from self.clean_stream(raw_text)


class OllamaCleaner(BaseLLMCleaner):
	"""Classify transcript intent through a local Ollama model."""

	def __init__(self, model_name: str = "llama3.2:1b") -> None:
		if not model_name.strip():
			raise ValueError("model_name must not be empty.")
		self.model_name = model_name
		self.system_prompt = WORD_FORMATTING_SYSTEM_PROMPT

	def process_transcript(self, raw_text: str) -> dict[str, str]:
		"""Return a validated command or dictation payload from Ollama."""
		if not isinstance(raw_text, str):
			raise TypeError("raw_text must be a string.")
		if not raw_text.strip():
			return {"type": "DICTATION", "text": ""}
		formatting_payload = self._known_formatting_payload(raw_text)
		if formatting_payload is not None:
			return formatting_payload

		try:
			response = ollama.chat(
				model=self.model_name,
				messages=[
					{"role": "system", "content": self.system_prompt},
					{"role": "user", "content": raw_text},
				],
				format=TranscriptPayload.model_json_schema(),
				options={"temperature": 0.0},
			)
			payload = TranscriptPayload.model_validate(json.loads(self._response_content(response)))
			return payload.model_dump(exclude_none=True, mode="json")
		except (Exception,):
			return {"type": "DICTATION", "text": raw_text.strip()}

	@staticmethod
	def _response_content(response: object) -> str:
		if isinstance(response, dict):
			message = response.get("message", {})
			return message.get("content", "") if isinstance(message, dict) else ""

		message = getattr(response, "message", None)
		return getattr(message, "content", "") if message is not None else ""

	@staticmethod
	def _known_formatting_payload(raw_text: str) -> dict[str, str] | None:
		"""Recognize direct Word formatting requests before LLM classification."""
		command = raw_text.lower()
		if re.search(r"\b(?:bullet|bullets|bullet point|bullet list)\b", command):
			return {"type": "COMMAND", "action": "TOGGLE_BULLETS"}
		if re.search(r"\b(?:bold|bolden)\b", command):
			return {"type": "COMMAND", "action": "BOLD"}
		if re.search(r"\b(?:italic|italicize|italics)\b", command):
			return {"type": "COMMAND", "action": "ITALIC"}
		if re.search(r"\b(?:underline|underlined)\b", command):
			return {"type": "COMMAND", "action": "UNDERLINE"}
		if re.search(r"\b(?:align|alignment)\s+(?:to\s+)?(?:the\s+)?left\b", command):
			return {"type": "COMMAND", "action": "ALIGN_LEFT"}
		if re.search(r"\b(?:center|centre|align\s+(?:to\s+)?(?:the\s+)?center|align\s+(?:to\s+)?(?:the\s+)?centre)\b", command):
			return {"type": "COMMAND", "action": "ALIGN_CENTER"}
		if re.search(r"\b(?:align|alignment)\s+(?:to\s+)?(?:the\s+)?right\b", command):
			return {"type": "COMMAND", "action": "ALIGN_RIGHT"}
		if re.search(r"\b(?:increase|enlarge|grow)\b.*\b(?:font|pound|text)(?:\s+size)?\b", command) or re.search(
			r"\bmake\b.*\b(?:font|pound|text)\b.*\b(?:bigger|larger)\b", command
		):
			return {"type": "COMMAND", "action": "FONT_INCREASE"}
		if re.search(r"\b(?:decrease|reduce|shrink)\b.*\b(?:font|pound|text)(?:\s+size)?\b", command) or re.search(
			r"\bmake\b.*\b(?:font|pound|text)\b.*\bsmaller\b", command
		):
			return {"type": "COMMAND", "action": "FONT_DECREASE"}
		return None


class MockLLMCleaner(BaseLLMCleaner):
	"""Deterministic cleaner used until a local LLM backend is configured."""

	def __init__(self, response_chunks: Iterable[str] | None = None) -> None:
		self._response_chunks = tuple(response_chunks or ("This is cleaned text.",))
		if any(not isinstance(chunk, str) for chunk in self._response_chunks):
			raise TypeError("response_chunks must contain only strings.")
		self.last_raw_text = ""

	def process_transcript(self, raw_text: str) -> dict[str, str]:
		"""Return configured deterministic text as a dictation payload."""
		if not isinstance(raw_text, str):
			raise TypeError("raw_text must be a string.")

		self.last_raw_text = raw_text
		if not raw_text:
			return {"type": "DICTATION", "text": ""}

		return {"type": "DICTATION", "text": "".join(self._response_chunks)}

	def clean_stream(self, raw_text: str) -> Generator[str, None, None]:
		"""Store the raw transcript and yield the configured response chunks."""
		payload = self.process_transcript(raw_text)
		if payload["text"]:
			yield from self._response_chunks
