import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

try:
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:  # pragma: no cover - optional GUI dependency may be absent in slim envs
    QApplication = None
    QMessageBox = None


class FakeLibraryDb:
    def __init__(self, images, fail_remove_ids=None):
        self.images = list(images)
        self.fail_remove_ids = set(fail_remove_ids or [])
        self.removed = []

    def get_image_count(self):
        return len(self.images)

    def get_all_images(self, limit=100, offset=0):
        return self.images[offset:offset + limit]

    def search_by_text(self, text_query, top_k=5):
        query = text_query.lower()
        matches = [
            image for image in self.images
            if query in image["path"].lower()
            or query in image.get("metadata", {}).get("name", "").lower()
        ]
        return matches[:top_k]

    def remove_image(self, image_id):
        if image_id in self.fail_remove_ids:
            raise RuntimeError("locked")
        self.removed.append(image_id)
        self.images = [image for image in self.images if image["id"] != image_id]


@unittest.skipUnless(QApplication is not None, "PySide6 is not available")
class ColorPanelRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _create_library_dialog(self, image_db):
        from src.ui.library_manager_dialog import LibraryManagerDialog

        with mock.patch("src.ui.library_manager_dialog.ThumbnailLoader.start"):
            dialog = LibraryManagerDialog(image_db)
        self.addCleanup(dialog.close)
        return dialog

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

    def test_agi_animation_preview_handles_extreme_aspect_ratio_frames(self):
        from src.ui.agi_camera_panel import AnimationPreview

        preview = AnimationPreview()
        preview.resize(400, 350)
        preview.set_frames([np.full((1, 10000, 3), 127, dtype=np.uint8)])

        pixmap = preview.pixmap()
        self.assertIsNotNone(pixmap)
        self.assertFalse(pixmap.isNull())

    def test_agi_export_buttons_follow_generated_state(self):
        from src.ai import Mesh3D
        from src.ui.agi_camera_panel import AGICameraPanel

        panel = AGICameraPanel()
        self.assertFalse(panel.export_model_btn.isEnabled())
        self.assertFalse(panel.export_gif_btn.isEnabled())
        self.assertFalse(panel.export_video_btn.isEnabled())

        mesh = Mesh3D(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            colors=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        )
        panel.set_mesh(mesh)
        self.assertTrue(panel.export_model_btn.isEnabled())
        self.assertFalse(panel.export_gif_btn.isEnabled())
        self.assertFalse(panel.export_video_btn.isEnabled())

        panel.set_animation([np.full((8, 8, 3), 127, dtype=np.uint8)])
        self.assertTrue(panel.export_gif_btn.isEnabled())
        self.assertTrue(panel.export_video_btn.isEnabled())

        panel.set_image(np.full((8, 8, 3), 127, dtype=np.uint8))
        self.assertFalse(panel.export_model_btn.isEnabled())
        self.assertFalse(panel.export_gif_btn.isEnabled())
        self.assertFalse(panel.export_video_btn.isEnabled())

    def test_agi_generation_controls_follow_image_state(self):
        from src.ui.agi_camera_panel import AGICameraPanel

        panel = AGICameraPanel()

        self.assertFalse(panel.generate_3d_btn.isEnabled())
        self.assertFalse(panel.generate_anim_btn.isEnabled())
        self.assertFalse(panel.point_mode_btn.isEnabled())
        self.assertFalse(panel.generate_object_3d_btn.isEnabled())

        panel.set_image_available(True)
        self.assertTrue(panel.generate_3d_btn.isEnabled())
        self.assertTrue(panel.generate_anim_btn.isEnabled())
        self.assertTrue(panel.point_mode_btn.isEnabled())
        self.assertFalse(panel.generate_object_3d_btn.isEnabled())

        panel.set_selection_result(True)
        self.assertTrue(panel.generate_object_3d_btn.isEnabled())

        panel.set_image_available(False)
        self.assertFalse(panel.generate_3d_btn.isEnabled())
        self.assertFalse(panel.generate_anim_btn.isEnabled())
        self.assertFalse(panel.point_mode_btn.isEnabled())
        self.assertFalse(panel.generate_object_3d_btn.isEnabled())

    def test_library_manager_selection_state_clears_detail_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "valid.png"
            cv2.imwrite(str(image_path), np.full((6, 8, 3), 127, dtype=np.uint8))
            image_db = FakeLibraryDb([
                {"id": "image-one", "path": str(image_path), "metadata": {"name": "valid"}}
            ])

            dialog = self._create_library_dialog(image_db)
            self.assertEqual(dialog.list_widget.count(), 1)
            self.assertFalse(dialog.btn_delete.isEnabled())
            self.assertEqual(dialog.lbl_filename.text(), "-")

            dialog.list_widget.setCurrentRow(0)
            self.app.processEvents()

            self.assertTrue(dialog.btn_delete.isEnabled())
            self.assertEqual(dialog.lbl_filename.text(), image_path.name)
            self.assertEqual(dialog.lbl_resolution.text(), "8 x 6")

            dialog.list_widget.clearSelection()
            self.app.processEvents()

            self.assertFalse(dialog.btn_delete.isEnabled())
            self.assertEqual(dialog.lbl_filename.text(), "-")
            self.assertEqual(dialog.lbl_resolution.text(), "-")
            self.assertEqual(dialog.img_preview.text(), "无预览")

    def test_library_manager_context_delete_targets_right_clicked_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.png"
            second_path = Path(tmp) / "second.png"
            cv2.imwrite(str(first_path), np.full((8, 8, 3), 63, dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            image_db = FakeLibraryDb([
                {"id": "first", "path": str(first_path), "metadata": {"name": "first"}},
                {"id": "second", "path": str(second_path), "metadata": {"name": "second"}},
            ])

            dialog = self._create_library_dialog(image_db)
            dialog.show()
            self.app.processEvents()
            dialog.list_widget.setCurrentRow(0)

            target = dialog.list_widget.item(1)
            dialog._select_context_item(target)
            with mock.patch("src.ui.library_manager_dialog.ThumbnailLoader.start"), \
                    mock.patch("src.ui.library_manager_dialog.QMessageBox.question", return_value=QMessageBox.Yes), \
                    mock.patch("src.ui.library_manager_dialog.QMessageBox.information"):
                dialog._on_delete_clicked()

            self.assertEqual(image_db.removed, ["second"])

    def test_library_manager_delete_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            good_path = Path(tmp) / "good.png"
            bad_path = Path(tmp) / "bad.png"
            cv2.imwrite(str(good_path), np.full((8, 8, 3), 63, dtype=np.uint8))
            cv2.imwrite(str(bad_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            image_db = FakeLibraryDb([
                {"id": "good", "path": str(good_path), "metadata": {"name": "good"}},
                {"id": "bad", "path": str(bad_path), "metadata": {"name": "bad"}},
            ], fail_remove_ids={"bad"})
            dialog = self._create_library_dialog(image_db)
            dialog.list_widget.selectAll()

            with mock.patch("src.ui.library_manager_dialog.ThumbnailLoader.start"), \
                    mock.patch("src.ui.library_manager_dialog.QMessageBox.question", return_value=QMessageBox.Yes), \
                    mock.patch("src.ui.library_manager_dialog.QMessageBox.warning") as warning:
                dialog._on_delete_clicked()

            self.assertEqual(image_db.removed, ["good"])
            warning.assert_called_once()

    def test_library_manager_open_missing_folder_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_db = FakeLibraryDb([])
            dialog = self._create_library_dialog(image_db)
            missing_path = Path(tmp) / "missing" / "image.png"

            with mock.patch("src.ui.library_manager_dialog.QMessageBox.warning") as warning:
                self.assertFalse(dialog._open_file_in_explorer(str(missing_path)))

            warning.assert_called_once()
            self.assertIn("无法打开文件位置", dialog.status_bar.text())


if __name__ == "__main__":
    unittest.main()
