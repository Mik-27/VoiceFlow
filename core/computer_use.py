"""OS-Atlas Computer Use Agent for autonomous GUI grounding and action execution."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Tuple

try:
	from PIL import ImageGrab
except ImportError:  # pragma: no cover
	ImageGrab = None  # type: ignore

try:
	import pyautogui
except ImportError:  # pragma: no cover
	pyautogui = None  # type: ignore

try:
	import torch
	from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:  # pragma: no cover
	torch = None  # type: ignore
	AutoModelForCausalLM = None  # type: ignore
	AutoProcessor = None  # type: ignore

from utils.logger import get_logger

if TYPE_CHECKING:
	from PIL.Image import Image


class OSAtlasAgent:
	"""GUI Computer Use Agent powered by OS-Atlas for desktop element grounding."""

	def __init__(
		self,
		model_path: str = "OS-Atlas/OS-Atlas-Base-7B",
		device: str = "cuda",
	) -> None:
		self.logger = get_logger("computer_use")
		self.device = device if (torch is not None and torch.cuda.is_available()) else "cpu"
		pyautogui.FAILSAFE = True  # Safety: Move mouse to top-left corner to abort

		self.logger.info("Initializing OSAtlasAgent on device: %s (model: %s)", self.device, model_path)

		if AutoProcessor is None or AutoModelForCausalLM is None:
			raise ImportError("transformers and torch packages are required to run OSAtlasAgent.")

		# Load OS Atlas model & processor
		self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
		torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
		self.model = AutoModelForCausalLM.from_pretrained(
			model_path,
			torch_dtype=torch_dtype,
			trust_remote_code=True,
		).to(self.device)
		self.logger.info("OSAtlasAgent model loaded successfully.")

	def capture_screen(self) -> Image:
		"""Capture the primary desktop display screenshot."""
		return ImageGrab.grab()

	def predict_click_coordinates(self, image: Image, instruction: str) -> Tuple[int, int]:
		"""Pass screenshot and user prompt to OS Atlas to predict (x, y) target coordinates."""
		# Format prompt for OS-Atlas grounding
		formatted_instruction = f"In this UI screenshot, what are the coordinates to {instruction}?"
		inputs = self.processor(images=image, text=formatted_instruction, return_tensors="pt").to(self.device)

		with torch.no_grad():
			outputs = self.model.generate(**inputs, max_new_tokens=128)

		response = self.processor.decode(outputs[0], skip_special_tokens=True)
		self.logger.debug("OS Atlas raw response: %s", response)

		return self._parse_coordinates(response, image.size)

	def _parse_coordinates(self, response: str, screen_size: Tuple[int, int]) -> Tuple[int, int]:
		"""Extract normalized coordinates from model response and convert to pixel coordinates."""
		screen_width, screen_height = screen_size

		# Check for bounding box format: [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2]
		bbox_match = re.search(r"\[\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\]", response)
		if bbox_match:
			c1, c2, c3, c4 = map(int, bbox_match.groups())
			# OS-Atlas format is typically [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
			# Taking averages across min/max pairs yields center point
			norm_x = (c2 + c4) / 2.0 if c2 <= 1000 and c4 <= 1000 else (c1 + c3) / 2.0
			norm_y = (c1 + c3) / 2.0 if c2 <= 1000 and c4 <= 1000 else (c2 + c4) / 2.0
		else:
			# Check for point format: [x, y] or (x, y) or "x, y"
			point_match = re.search(r"[\[\(]?\s*(\d+)\s*,\s*(\d+)\s*[\]\)]?", response)
			if point_match:
				norm_x, norm_y = map(float, point_match.groups())
			else:
				# Fallback: find all numbers in the response
				numbers = [int(n) for n in re.findall(r"\d+", response)]
				if len(numbers) >= 4:
					norm_x = (numbers[1] + numbers[3]) / 2.0
					norm_y = (numbers[0] + numbers[2]) / 2.0
				elif len(numbers) >= 2:
					norm_x, norm_y = float(numbers[0]), float(numbers[1])
				else:
					raise ValueError(f"Could not parse click coordinates from OS Atlas response: {response!r}")

		# Convert normalized coordinates [0, 1000] (or [0, 1]) to screen pixel coordinates
		scale_x = screen_width / 1000.0 if norm_x > 1.0 else screen_width
		scale_y = screen_height / 1000.0 if norm_y > 1.0 else screen_height

		x_pixel = max(0, min(screen_width - 1, int(norm_x * scale_x)))
		y_pixel = max(0, min(screen_height - 1, int(norm_y * scale_y)))

		return x_pixel, y_pixel

	def execute_action(self, goal_description: str) -> bool:
		"""Perform closed-loop perception and mouse interaction."""
		self.logger.info("Executing computer action for goal: %r", goal_description)
		try:
			# 1. Capture current display
			screenshot = self.capture_screen()

			# 2. Predict target click point
			x, y = self.predict_click_coordinates(screenshot, goal_description)
			self.logger.info("Target coordinates resolved: (%d, %d)", x, y)

			# 3. Execute native mouse click
			pyautogui.moveTo(x, y, duration=0.2)
			pyautogui.click()

			time.sleep(0.1)
			return True
		except Exception as e:
			self.logger.exception("[OSAtlasAgent] Execution failed: %s", e)
			return False
