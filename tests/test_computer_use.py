from core import computer_use
from core.computer_use import CUAgent


def test_default_model_path_uses_gui_actor():
    assert CUAgent.DEFAULT_MODEL_PATH == "microsoft/GUI-Actor-3B-Qwen2.5-VL"


def test_gui_actor_import_failure_does_not_hide_torch():
    assert computer_use.torch is not None
