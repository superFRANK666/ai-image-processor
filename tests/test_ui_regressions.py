import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

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

    def test_preset_submission_does_not_override_external_busy_state(self):
        from PySide6.QtCore import QEventLoop, QTimer
        from src.ui.color_grading_panel import ColorGradingPanel

        panel = ColorGradingPanel()

        def mark_busy(_text):
            panel.apply_btn.setText("分析中...")
            panel.apply_btn.setEnabled(False)

        panel.text_input_submitted.connect(mark_busy)
        panel.preset_combo.setCurrentText("电影感")

        loop = QEventLoop()
        QTimer.singleShot(650, loop.quit)
        loop.exec()

        self.assertEqual(panel.apply_btn.text(), "分析中...")
        self.assertFalse(panel.apply_btn.isEnabled())

    def test_color_panel_can_reflect_missing_image_state(self):
        from src.ui.color_grading_panel import ColorGradingPanel

        panel = ColorGradingPanel()
        panel.set_image_available(False)

        self.assertFalse(panel.apply_btn.isEnabled())
        self.assertFalse(panel.preset_combo.isEnabled())
        self.assertFalse(panel.exposure_slider.isEnabled())

        panel.set_image_available(True)

        self.assertTrue(panel.apply_btn.isEnabled())
        self.assertTrue(panel.preset_combo.isEnabled())
        self.assertTrue(panel.exposure_slider.isEnabled())

    def test_thumbnail_size_never_rounds_down_to_zero(self):
        from src.ui.ui_utils import fit_thumbnail_size, fit_within_size

        self.assertEqual(fit_thumbnail_size(10000, 1, 120), (120, 1))
        self.assertEqual(fit_thumbnail_size(1, 10000, 120), (1, 120))
        self.assertEqual(fit_within_size(10000, 1, 400, 350), (400, 1))
        self.assertEqual(fit_within_size(1, 10000, 400, 350), (1, 350))

        with self.assertRaises(ValueError):
            fit_thumbnail_size(0, 100, 120)
        with self.assertRaises(ValueError):
            fit_within_size(100, 100, 0, 120)

    def test_missing_thumbnail_sources_do_not_leave_grid_gaps(self):
        from src.ui.image_library_panel import ImageLibraryPanel
        from src.ui.image_picker_dialog import ImagePickerDialog

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "valid.png"
            missing_path = Path(tmp) / "missing.png"
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 127, dtype=np.uint8))

            results = [
                {"id": "missing", "path": str(missing_path), "metadata": {"name": "missing"}},
                {"id": "valid", "path": str(image_path), "metadata": {"name": "valid"}},
            ]

            panel = ImageLibraryPanel()
            shown_count = panel.show_search_results(results)
            self.assertEqual(shown_count, 1)
            self.assertEqual(len(panel._thumbnails), 1)
            self.assertIsNotNone(panel.thumbnail_layout.itemAtPosition(0, 0))

            dialog = ImagePickerDialog()
            dialog._show_thumbnails(results)
            self.assertEqual(len(dialog._thumbnails), 1)
            self.assertIsNotNone(dialog.thumbnail_layout.itemAtPosition(0, 0))

    def test_agi_exports_warn_when_animation_is_missing(self):
        from PySide6.QtWidgets import QMessageBox
        from src.ui.agi_camera_panel import AGICameraPanel

        panel = AGICameraPanel()
        with mock.patch.object(QMessageBox, "warning") as warning:
            panel._export_gif()
            panel._export_video()

        self.assertEqual(warning.call_count, 2)


if __name__ == "__main__":
    unittest.main()
