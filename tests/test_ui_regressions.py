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

    def test_thumbnail_size_never_rounds_down_to_zero(self):
        from src.ui.ui_utils import fit_thumbnail_size

        self.assertEqual(fit_thumbnail_size(10000, 1, 120), (120, 1))
        self.assertEqual(fit_thumbnail_size(1, 10000, 120), (1, 120))

        with self.assertRaises(ValueError):
            fit_thumbnail_size(0, 100, 120)


if __name__ == "__main__":
    unittest.main()
