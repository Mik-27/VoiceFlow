from pynput.keyboard import Controller, Key


class WordFormatter:
    """Handles native OS macro/hotkey commands for Microsoft Word on Windows/Linux."""

    def __init__(self):
        self.keyboard = Controller()

    def execute_formatting(self, command: str) -> bool:
        """Execute a native Word hotkey sequence for a recognized command."""
        cmd = command.upper()

        if cmd == "TOGGLE_BULLETS":
            with self.keyboard.pressed(Key.ctrl), self.keyboard.pressed(Key.shift):
                self.keyboard.tap("l")
            return True

        if cmd == "FONT_INCREASE":
            with self.keyboard.pressed(Key.ctrl), self.keyboard.pressed(Key.shift):
                self.keyboard.tap(".")
            return True

        if cmd == "FONT_DECREASE":
            with self.keyboard.pressed(Key.ctrl), self.keyboard.pressed(Key.shift):
                self.keyboard.tap(",")
            return True

        if cmd == "BOLD":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("b")
            return True

        if cmd == "ITALIC":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("i")
            return True

        if cmd == "UNDERLINE":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("u")
            return True

        if cmd == "ALIGN_LEFT":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("l")
            return True

        if cmd == "ALIGN_CENTER":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("e")
            return True

        if cmd == "ALIGN_RIGHT":
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.tap("r")
            return True

        return False