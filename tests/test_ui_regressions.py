import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - optional GUI dependency may be absent in slim envs
    QApplication = None


@unittest.skipUnless(QApplication is not None, "PySide6 is not available")
class ColorPanelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_programmatic_reset_can_be_silent_but_button_reset_emits(self):
        from src.ui.color_grading_panel import ColorGradingPanel

        panel = ColorGradingPanel()
        emitted = []
        panel.params_changed.connect(emitted.append)

        panel.reset_params(emit_change=False)
        self.assertEqual(emitted, [])

        panel.reset_btn.click()
        self.assertEqual(len(emitted), 1)


if __name__ == "__main__":
    unittest.main()
