"""OS-Atlas Computer Use Agent for autonomous GUI grounding and action execution."""

from __future__ import annotations

from importlib import import_module
import os
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
except ImportError:  # pragma: no cover
	torch = None  # type: ignore

try:
	from transformers import AutoProcessor, BitsAndBytesConfig
	from transformers.utils import logging as transformers_logging
	from huggingface_hub import snapshot_download
	from huggingface_hub.utils import enable_progress_bars
except ImportError:  # pragma: no cover
	AutoModelForCausalLM = None  # type: ignore
	AutoProcessor = None  # type: ignore
	BitsAndBytesConfig = None  # type: ignore
	transformers_logging = None  # type: ignore
	snapshot_download = None  # type: ignore
	enable_progress_bars = None  # type: ignore

try:
	grounding_system_message = import_module("gui_actor.constants").grounding_system_message
	gui_actor_inference = import_module("gui_actor.inference").inference
	Qwen2_5_VLForConditionalGenerationWithPointer = import_module(
		"gui_actor.modeling_qwen25vl"
	).Qwen2_5_VLForConditionalGenerationWithPointer
except ImportError:  # pragma: no cover
	grounding_system_message = None  # type: ignore
	gui_actor_inference = None  # type: ignore
	Qwen2_5_VLForConditionalGenerationWithPointer = None  # type: ignore

from utils.logger import get_logger

if TYPE_CHECKING:
	from PIL.Image import Image


class CUAgent:
	"""GUI Computer Use Agent for desktop element grounding."""

	DEFAULT_MODEL_PATH = "microsoft/GUI-Actor-3B-Qwen2.5-VL"

	def __init__(
		self,
		model_path: str = DEFAULT_MODEL_PATH,
		device: str = "cuda",
		load_in_4bit: bool = False,
		token: str | None = None,
	) -> None:
		self.logger = get_logger("computer_use")
		cuda_available = torch is not None and torch.cuda.is_available()
		self.device = device if (device == "cuda" and cuda_available) else "cpu"
		self.logger.debug("CUDA available: %s", cuda_available)
		pyautogui.FAILSAFE = True  # Safety: Move mouse to top-left corner to abort
		self.model_path = model_path

		self.logger.info("Initializing CUAgent on device: %s (model: %s)", self.device, self.model_path)

		if torch is None or AutoProcessor is None:
			raise ImportError("transformers and torch packages are required to run CUAgent.")
		if self.device != "cuda":
			raise RuntimeError(
				"GUI-Actor requires CUDA. Run VoiceFlow with the CUDA-enabled project interpreter: "
				".\\.venv\\Scripts\\python.exe .\\main.py"
			)
		if Qwen2_5_VLForConditionalGenerationWithPointer is None or gui_actor_inference is None:
			raise ImportError(
				"GUI-Actor's pointer-head package is not installed. It requires Python 3.10-3.12 and "
				"Transformers 4.51.3; the current Python 3.14 / Transformers 5.17 environment is unsupported."
			)

		# Enable explicit progress bars for download feedback
		if enable_progress_bars is not None:
			enable_progress_bars()
		if transformers_logging is not None:
			transformers_logging.enable_progress_bar()

		hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

		# Pre-download / verify repository snapshot with visual progress bar
		if snapshot_download is not None:
			self.logger.info("Checking/downloading repository snapshot for %s...", model_path)
			snapshot_download(
				repo_id=model_path,
				token=hf_token,
			)

		# Load OS Atlas model & processor
		self.logger.info("Loading processor from %s...", model_path)
		self.processor = AutoProcessor.from_pretrained(
			model_path,
			trust_remote_code=True,
			token=hf_token,
		)
		self.logger.info("Processor loaded successfully.")

		torch_dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
			torch.float16 if self.device == "cuda" else torch.float32
		)
		self.logger.info("Loading model weights into %s memory (dtype: %s)...", self.device, torch_dtype)

		# Quantization config if requested or on low-VRAM GPUs
		quantization_config = None
		if load_in_4bit and BitsAndBytesConfig is None:
			raise ImportError("4-bit GUI-Actor loading requires bitsandbytes. Install it with: pip install bitsandbytes")
		if load_in_4bit and self.device == "cuda":
			self.logger.info("Enabling 4-bit quantization for low-VRAM GPU.")
			quantization_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_compute_dtype=torch_dtype,
				bnb_4bit_quant_type="nf4",
			)

		model_kwargs: dict[str, object] = {
			"trust_remote_code": True,
			"token": hf_token,
			"attn_implementation": "sdpa",
		}

		if quantization_config is not None:
			model_kwargs["quantization_config"] = quantization_config
			model_kwargs["device_map"] = "cuda:0"
		elif self.device == "cuda":
			model_kwargs["torch_dtype"] = torch_dtype
			model_kwargs["device_map"] = "cuda:0"
		else:
			model_kwargs["torch_dtype"] = torch_dtype

		self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(model_path, **model_kwargs).eval()
		if "device_map" not in model_kwargs and self.device != "cpu":
			self.model = self.model.to(self.device)

		self.logger.info("CUAgent model loaded successfully.")

	def capture_screen(self) -> Image:
		"""Capture the primary desktop display screenshot."""
		return ImageGrab.grab()

	def predict_click_coordinates(self, image: Image, instruction: str) -> Tuple[int, int]:
		"""Use GUI-Actor's pointer head to resolve an instruction to screen pixels."""
		messages = [
			{
				"role": "system",
				"content": [{"type": "text", "text": grounding_system_message}],
			},
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image},
					{"type": "text", "text": instruction.strip().rstrip(".")},
				],
			},
		]
		prediction = gui_actor_inference(
			messages,
			self.model,
			self.processor.tokenizer,
			self.processor,
			use_placeholder=True,
			topk=1,
		)
		points = prediction.get("topk_points") or []
		if not points:
			raise ValueError(f"GUI-Actor did not return a click point: {prediction.get('output_text')!r}")

		normalized_x, normalized_y = points[0]
		screen_width, screen_height = image.size
		return (
			max(0, min(screen_width - 1, round(normalized_x * screen_width))),
			max(0, min(screen_height - 1, round(normalized_y * screen_height))),
		)

	def _parse_coordinates(self, response: str, screen_size: Tuple[int, int]) -> Tuple[int, int]:
		"""Extract normalized coordinates from model response and convert to pixel coordinates."""
		screen_width, screen_height = screen_size

		# Check for coordinate pairs like (x1, y1), (x2, y2) or [x1, y1, x2, y2]
		bbox_matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", response)
		if len(bbox_matches) >= 2:
			x1, y1 = map(float, bbox_matches[0])
			x2, y2 = map(float, bbox_matches[1])
			norm_x = (x1 + x2) / 2.0
			norm_y = (y1 + y2) / 2.0
		else:
			# Check for single point format: (x, y) or [x, y]
			point_match = re.search(r"[\[\(]?\s*(\d+)\s*,\s*(\d+)\s*[\]\)]?", response)
			if point_match:
				norm_x, norm_y = map(float, point_match.groups())
			else:
				# Fallback: find all numbers in the response
				numbers = [float(n) for n in re.findall(r"\d+", response)]
				if len(numbers) >= 4:
					norm_x = (numbers[0] + numbers[2]) / 2.0
					norm_y = (numbers[1] + numbers[3]) / 2.0
				elif len(numbers) >= 2:
					norm_x, norm_y = numbers[0], numbers[1]
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
			self.logger.exception("[CUAgent] Execution failed: %s", e)
			return False
