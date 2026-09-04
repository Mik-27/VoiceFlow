"""Transcript-cleaning language model interfaces and prompts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable


SYSTEM_PROMPT = """You clean speech-to-text transcripts for direct insertion into the user's active application.

Rules:
1. Return only the cleaned transcript. Do not add explanations, introductions, labels, or quotation marks.
2. Remove spoken disfluencies and filler words such as um, uh, er, like, you know, and repeated words when they are not meaningful.
3. Preserve the speaker's original meaning, wording, intent, names, numbers, and technical terms. Never invent, summarize, or answer the transcript.
4. Apply normal capitalization and punctuation. Use sentence breaks and paragraph breaks when the transcript clearly indicates them.
5. Keep the original language and tone. Do not translate, expand abbreviations, or make the text more formal unless required for basic grammar.
6. Preserve commands, code, URLs, email addresses, and special formatting exactly whenever they are recognizable.
7. If the input is already clear, return it with only necessary formatting corrections.
8. If the input is empty or contains no meaningful words, return an empty response.
"""


class BaseLLMCleaner(ABC):
	"""Stream cleaned transcript text from a local language model."""

	@abstractmethod
	def clean_text(self, raw_text: str) -> Generator[str, None, None]:
		"""Yield cleaned text chunks for a raw transcript."""


class MockLLMCleaner(BaseLLMCleaner):
	"""Deterministic cleaner used until a local LLM backend is configured."""

	def __init__(self, response_chunks: Iterable[str] | None = None) -> None:
		self._response_chunks = tuple(response_chunks or ("This is cleaned text.",))
		if any(not isinstance(chunk, str) for chunk in self._response_chunks):
			raise TypeError("response_chunks must contain only strings.")
		self.last_raw_text = ""

	def clean_text(self, raw_text: str) -> Generator[str, None, None]:
		"""Store the raw transcript and yield the configured response chunks."""
		if not isinstance(raw_text, str):
			raise TypeError("raw_text must be a string.")

		self.last_raw_text = raw_text
		if not raw_text:
			return

		yield from self._response_chunks
