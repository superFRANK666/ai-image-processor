import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

try:
    from PySide6.QtWidgets import QApplication, QMessageBox
except ImportError:  # pragma: no cover - optional GUI dependency may be absent in slim envs
    QApplication = None
    QMessageBox = None


class FakeLibraryDb:
    def __init__(
            self,
            images,
            fail_remove_ids=None,
            groups=None,
            fail_search=False,
            fail_update_ids=None,
            fail_group_ops=None,
            fail_add_names=None):
        self.images = list(images)
        self.fail_remove_ids = set(fail_remove_ids or [])
        self.fail_update_ids = set(fail_update_ids or [])
        self.fail_group_ops = set(fail_group_ops or [])
        self.fail_add_names = set(fail_add_names or [])
        self.fail_search = fail_search
        self.groups = list(groups or ["默认"])
        self.removed = []
        self.added_images = []
        self.added_groups = []
        self.renamed_groups = []
        self.deleted_groups = []

    def get_image_count(self):
        return len(self.images)

    def get_all_images(self, limit=100, offset=0):
        return self.images[offset:offset + limit]

    def get_images_by_group(self, _group, limit=50):
        return self.images[:limit]

    def search_by_text(self, text_query, top_k=5):
        if self.fail_search:
            raise RuntimeError("index offline")
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

    def update_image_metadata(self, image_id, new_metadata):
        if image_id in self.fail_update_ids:
            raise RuntimeError("metadata locked")
        for image in self.images:
            if image["id"] == image_id:
                image.setdefault("metadata", {}).update(new_metadata)
                return

    def get_groups(self):
        return list(self.groups)

    def add_group(self, group_name):
        if "add" in self.fail_group_ops:
            raise RuntimeError("cannot add")
        self.added_groups.append(group_name)
        if group_name not in self.groups:
            self.groups.append(group_name)

    def delete_group(self, group_name):
        if "delete" in self.fail_group_ops:
            raise RuntimeError("cannot delete")
        self.deleted_groups.append(group_name)
        self.groups = [group for group in self.groups if group != group_name]

    def rename_group(self, old_name, new_name):
        if "rename" in self.fail_group_ops:
            raise RuntimeError("cannot rename")
        self.renamed_groups.append((old_name, new_name))
        self.groups = [new_name if group == old_name else group for group in self.groups]

    def add_image(self, path, group="默认"):
        if Path(path).name in self.fail_add_names:
            raise RuntimeError("decode failed")
        self.added_images.append((str(path), group))


class FakeProgress:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class FakeWorker:
    def __init__(self, stop_requested=False):
        self._stop_requested = stop_requested
        self.progress = FakeProgress()

    @property
    def stop_requested(self):
        return self._stop_requested


class FakeStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, timeout=0):
        self.messages.append((message, timeout))


class FakeProgressBar:
    def __init__(self):
        self.hidden = False
        self.shown = False
        self.ranges = []

    def hide(self):
        self.hidden = True
        self.shown = False

    def show(self):
        self.shown = True
        self.hidden = False

    def setRange(self, minimum, maximum):
        self.ranges.append((minimum, maximum))


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.text = "应用"

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setText(self, text):
        self.text = text


class FakeColorPanel:
    def __init__(self):
        self.apply_btn = FakeButton()
        self.params = None

    def set_params(self, params):
        self.params = params


class FakeParser:
    def __init__(self):
        self.calls = []

    def parse_async(self, *args, **kwargs):
        self.calls.append((args, kwargs))


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

    def test_image_picker_empty_state_disables_selection_actions(self):
        from src.ui.image_picker_dialog import ImagePickerDialog

        dialog = ImagePickerDialog()
        self.addCleanup(dialog.close)

        self.assertFalse(dialog.ok_btn.isEnabled())
        self.assertFalse(dialog.clear_btn.isEnabled())
        self.assertFalse(dialog.tab_widget.isTabEnabled(1))
        self.assertFalse(dialog.search_btn.isEnabled())
        self.assertEqual(dialog.status_label.text(), "图像库未初始化")

        with mock.patch("src.ui.image_picker_dialog.QMessageBox.information") as information:
            dialog._accept_selection()

        information.assert_called_once()

    def test_image_picker_browse_selection_updates_actions_and_display(self):
        from src.ui.image_picker_dialog import ImagePickerDialog

        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.png"
            second_path = Path(tmp) / "second.png"
            cv2.imwrite(str(first_path), np.full((8, 8, 3), 63, dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full((8, 8, 3), 127, dtype=np.uint8))

            dialog = ImagePickerDialog(multi_select=True)
            self.addCleanup(dialog.close)
            with mock.patch(
                "src.ui.image_picker_dialog.QFileDialog.getOpenFileNames",
                return_value=([str(first_path), str(second_path)], "")
            ):
                dialog._on_browse_files()

            self.assertTrue(dialog.ok_btn.isEnabled())
            self.assertTrue(dialog.clear_btn.isEnabled())
            self.assertIn("first.png", dialog.selection_list.text())
            self.assertIn("second.png", dialog.selection_list.text())

            dialog._clear_selection()
            self.assertFalse(dialog.ok_btn.isEnabled())
            self.assertFalse(dialog.clear_btn.isEnabled())
            self.assertEqual(dialog.selection_list.text(), "无")

    def test_image_picker_library_refresh_and_search_statuses(self):
        from src.ui.image_picker_dialog import ImagePickerDialog

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "valid.png"
            missing_path = Path(tmp) / "missing.png"
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            image_db = FakeLibraryDb([
                {"id": "valid", "path": str(image_path), "metadata": {"name": "valid"}},
                {"id": "missing", "path": str(missing_path), "metadata": {"name": "missing"}},
            ])

            dialog = ImagePickerDialog(image_db)
            self.addCleanup(dialog.close)

            self.assertTrue(dialog.tab_widget.isTabEnabled(1))
            self.assertEqual(len(dialog._thumbnails), 1)
            self.assertEqual(dialog.status_label.text(), "图像库: 2 张图片，当前显示 1 张")

            dialog.search_input.setText("missing")
            dialog._on_search()

            self.assertEqual(len(dialog._thumbnails), 0)
            self.assertEqual(dialog.status_label.text(), "未找到匹配 \"missing\" 的图片")

    def test_image_picker_single_select_replaces_previous_thumbnail_choice(self):
        from src.ui.image_picker_dialog import ImagePickerDialog

        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.png"
            second_path = Path(tmp) / "second.png"
            cv2.imwrite(str(first_path), np.full((8, 8, 3), 63, dtype=np.uint8))
            cv2.imwrite(str(second_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            image_db = FakeLibraryDb([
                {"id": "first", "path": str(first_path), "metadata": {"name": "first"}},
                {"id": "second", "path": str(second_path), "metadata": {"name": "second"}},
            ])

            dialog = ImagePickerDialog(image_db, multi_select=False)
            self.addCleanup(dialog.close)
            first_thumb, second_thumb = dialog._thumbnails

            dialog._on_thumbnail_clicked(first_thumb.image_path, True)
            dialog._on_thumbnail_clicked(second_thumb.image_path, True)

            self.assertEqual(dialog.get_selected_paths(), [str(second_path)])
            self.assertFalse(first_thumb.is_selected())
            self.assertTrue(second_thumb.is_selected())

    def test_image_library_unavailable_database_disables_library_actions(self):
        from src.ui.image_library_panel import ImageLibraryPanel

        panel = ImageLibraryPanel()
        self.addCleanup(panel.close)

        self.assertFalse(panel.search_input.isEnabled())
        self.assertFalse(panel.search_btn.isEnabled())
        self.assertFalse(panel.refresh_btn.isEnabled())
        self.assertFalse(panel.rebuild_btn.isEnabled())
        self.assertFalse(panel.new_group_btn.isEnabled())
        self.assertFalse(panel.group_combo.isEnabled())
        self.assertTrue(panel.import_btn.isEnabled())
        self.assertEqual(panel.status_label.text(), "图像库未初始化")
        self.assertEqual(panel.search_input.placeholderText(), "图像库加载后可搜索...")

        panel.set_database(FakeLibraryDb([]))

        self.assertTrue(panel.search_input.isEnabled())
        self.assertTrue(panel.search_btn.isEnabled())
        self.assertTrue(panel.refresh_btn.isEnabled())
        self.assertTrue(panel.rebuild_btn.isEnabled())
        self.assertTrue(panel.new_group_btn.isEnabled())
        self.assertTrue(panel.group_combo.isEnabled())
        self.assertEqual(panel.search_input.placeholderText(), "搜索图像...")

    def test_image_library_group_names_are_validated_and_normalized(self):
        from src.ui.image_library_panel import ImageLibraryPanel

        image_db = FakeLibraryDb([], groups=["默认", "已有"])
        panel = ImageLibraryPanel(image_db)
        self.addCleanup(panel.close)
        panel.refresh()

        with mock.patch("src.ui.image_library_panel.QInputDialog.getText", return_value=("  已有  ", True)), \
                mock.patch("src.ui.image_library_panel.QMessageBox.warning") as warning:
            panel._on_new_group()

        warning.assert_called_once()
        self.assertEqual(image_db.added_groups, [])
        self.assertEqual(panel.status_label.text(), "分组已存在: 已有")

        with mock.patch("src.ui.image_library_panel.QInputDialog.getText", return_value=("  新分组  ", True)):
            panel._on_new_group()

        self.assertIn("新分组", image_db.added_groups)
        self.assertEqual(panel.group_combo.currentText(), "新分组")

    def test_image_library_rename_group_validates_duplicates_and_strips_name(self):
        from src.ui.image_library_panel import ImageLibraryPanel

        image_db = FakeLibraryDb([], groups=["默认", "旧名", "已有"])
        panel = ImageLibraryPanel(image_db)
        self.addCleanup(panel.close)
        panel.refresh()

        with mock.patch("src.ui.image_library_panel.QInputDialog.getText", return_value=("已有", True)), \
                mock.patch("src.ui.image_library_panel.QMessageBox.warning") as warning:
            panel._rename_group("旧名")

        warning.assert_called_once()
        self.assertEqual(image_db.renamed_groups, [])

        with mock.patch("src.ui.image_library_panel.QInputDialog.getText", return_value=("  新名  ", True)):
            panel._rename_group("旧名")

        self.assertEqual(image_db.renamed_groups, [("旧名", "新名")])
        self.assertIn("新名", [panel.group_combo.itemText(i) for i in range(panel.group_combo.count())])

    def test_image_library_search_and_delete_failures_are_reported(self):
        from src.ui.image_library_panel import ImageLibraryPanel

        search_panel = ImageLibraryPanel(FakeLibraryDb([], fail_search=True))
        self.addCleanup(search_panel.close)
        search_panel.search_input.setText("风景")
        search_panel._on_search()
        self.assertEqual(search_panel.status_label.text(), "搜索失败: index offline")

        delete_panel = ImageLibraryPanel(FakeLibraryDb([], fail_remove_ids={"bad"}))
        self.addCleanup(delete_panel.close)
        with mock.patch("src.ui.image_library_panel.QMessageBox.question", return_value=QMessageBox.Yes), \
                mock.patch("src.ui.image_library_panel.QMessageBox.warning") as warning:
            delete_panel._delete_image("bad")

        warning.assert_called_once()
        self.assertEqual(delete_panel.status_label.text(), "删除失败")

    def test_image_library_import_allows_default_group_and_validates_new_group(self):
        from src.ui.image_library_panel import ImageLibraryPanel

        image_db = FakeLibraryDb([], groups=["默认"])
        panel = ImageLibraryPanel(image_db)
        self.addCleanup(panel.close)
        emitted = []
        panel.import_requested.connect(lambda files, group: emitted.append((files, group)))

        with mock.patch("src.ui.image_library_panel.pick_images", return_value=["image.png"]), \
                mock.patch("src.ui.image_library_panel.QInputDialog.getItem", return_value=("默认", True)):
            panel._on_import_btn_clicked()

        self.assertEqual(emitted, [(["image.png"], "默认")])

        with mock.patch("src.ui.image_library_panel.pick_images", return_value=["image.png"]), \
                mock.patch("src.ui.image_library_panel.QInputDialog.getItem", return_value=("全部", True)), \
                mock.patch("src.ui.image_library_panel.QMessageBox.warning") as warning:
            panel._on_import_btn_clicked()

        warning.assert_called_once()
        self.assertEqual(len(emitted), 1)

    def test_image_library_thumbnail_rename_strips_empty_and_reports_failures(self):
        from src.ui.image_library_panel import ImageLibraryPanel, ImageThumbnailWidget

        image_db = FakeLibraryDb([
            {"id": "ok", "path": "ok.png", "metadata": {"name": "old"}},
            {"id": "locked", "path": "locked.png", "metadata": {"name": "old"}},
        ], fail_update_ids={"locked"})
        panel = ImageLibraryPanel(image_db)
        self.addCleanup(panel.close)

        panel._on_rename_requested_from_thumb("ok", "old", "  new  ")
        self.assertEqual(image_db.images[0]["metadata"]["name"], "new")

        with mock.patch("src.ui.image_library_panel.QMessageBox.warning") as warning:
            panel._on_rename_requested_from_thumb("locked", "old", "new")

        warning.assert_called_once()
        self.assertEqual(panel.status_label.text(), "重命名失败")

        with tempfile.TemporaryDirectory() as tmp:
            thumb_path = Path(tmp) / "thumb.png"
            cv2.imwrite(str(thumb_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            thumbnail = ImageThumbnailWidget(str(thumb_path), name="old", img_id="ok")
            self.addCleanup(thumbnail.close)
            emitted = []
            thumbnail.rename_requested.connect(lambda *args: emitted.append(args))
            with mock.patch("src.ui.image_library_panel.QInputDialog.getText", return_value=("   ", True)):
                thumbnail._start_rename()

            self.assertEqual(emitted, [("ok", "old", "")])
            self.assertEqual(thumbnail.img_name, "old")

    def test_batch_operation_result_formats_user_summary(self):
        from src.ui.main_window import BatchOperationResult

        result = BatchOperationResult(total=4, success=2, canceled=True)
        result.add_failure("bad-one.png", RuntimeError("decode failed"))
        result.add_failure("bad-two.png", RuntimeError("missing metadata"))

        self.assertEqual(result.failed_count, 2)
        self.assertTrue(result.has_issues)
        self.assertEqual(result.status_message("导入"), "导入: 成功 2/4 张，失败 2 张，已取消")
        self.assertIn("失败详情:", result.detail_message("导入"))
        self.assertIn("bad-one.png: decode failed", result.detail_message("导入"))

    def test_main_window_batch_import_and_index_collect_failures(self):
        from src.ui.main_window import MainWindow

        image_db = FakeLibraryDb([], fail_add_names={"bad.png"})
        fake_window = SimpleNamespace(image_db=image_db)
        worker = FakeWorker()

        import_result = MainWindow._run_import_batch(
            fake_window,
            ["good.png", "bad.png"],
            "默认",
            worker
        )

        self.assertEqual(import_result.total, 2)
        self.assertEqual(import_result.success, 1)
        self.assertEqual(import_result.failed_count, 1)
        self.assertEqual(worker.progress.values, [1, 2])
        self.assertEqual(image_db.added_images, [("good.png", "默认")])

        index_result = MainWindow._run_index_batch(
            fake_window,
            [Path("good.png"), Path("bad.png")],
            FakeWorker()
        )

        self.assertEqual(index_result.success, 1)
        self.assertEqual(index_result.failed_count, 1)

    def test_main_window_batch_stops_cleanly_when_worker_is_canceled(self):
        from src.ui.main_window import MainWindow

        image_db = FakeLibraryDb([])
        fake_window = SimpleNamespace(image_db=image_db)

        result = MainWindow._run_import_batch(
            fake_window,
            ["first.png", "second.png"],
            "默认",
            FakeWorker(stop_requested=True)
        )

        self.assertTrue(result.canceled)
        self.assertEqual(result.success, 0)
        self.assertEqual(image_db.added_images, [])

    def test_main_window_supported_folder_scan_is_unique_and_sorted(self):
        from src.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.PNG").write_bytes(b"data")
            (root / "a.jpg").write_bytes(b"data")
            (root / "notes.txt").write_text("skip")

            files = MainWindow._collect_supported_image_files(object(), str(root))

            self.assertEqual([path.name for path in files], ["a.jpg", "b.PNG"])

            with self.assertRaises(ValueError):
                MainWindow._collect_supported_image_files(object(), str(root / "missing"))

    def test_main_window_batch_completion_uses_visible_feedback(self):
        from src.ui.main_window import BatchOperationResult, MainWindow

        fake_window = SimpleNamespace(statusbar=FakeStatusBar())
        success = BatchOperationResult(total=1, success=1)

        with mock.patch("src.ui.main_window.QMessageBox.information") as information:
            MainWindow._show_batch_completion(
                fake_window,
                "导入",
                success,
                show_success_dialog=True
            )

        information.assert_called_once()
        self.assertEqual(fake_window.statusbar.messages[-1], ("导入: 成功 1/1 张", 3000))

        failed = BatchOperationResult(total=2, success=1)
        failed.add_failure("bad.png", RuntimeError("decode failed"))
        with mock.patch("src.ui.main_window.QMessageBox.warning") as warning:
            MainWindow._show_batch_completion(fake_window, "索引", failed)

        warning.assert_called_once()
        self.assertEqual(fake_window.statusbar.messages[-1], ("索引: 成功 1/2 张，失败 1 张", 5000))

    def test_processing_error_rolls_back_pending_grading_history(self):
        from src.ui.main_window import MainWindow

        calls = []
        fake_window = SimpleNamespace(
            progress_bar=FakeProgressBar(),
            statusbar=FakeStatusBar(),
            _history_stack=["before"],
            _grading_history_pending=True,
            _pending_params=object(),
            _update_action_states=lambda: calls.append("updated"),
        )
        fake_window._rollback_pending_grading = lambda: MainWindow._rollback_pending_grading(fake_window)

        with mock.patch("src.ui.main_window.QMessageBox.warning") as warning:
            MainWindow._on_processing_error(fake_window, "boom")

        warning.assert_called_once()
        self.assertTrue(fake_window.progress_bar.hidden)
        self.assertEqual(fake_window._history_stack, [])
        self.assertFalse(fake_window._grading_history_pending)
        self.assertIsNone(fake_window._pending_params)
        self.assertEqual(calls, ["updated"])
        self.assertEqual(fake_window.statusbar.messages[-1], ("处理失败: boom", 5000))

    def test_processing_error_preserves_history_without_pending_grading(self):
        from src.ui.main_window import MainWindow

        fake_window = SimpleNamespace(
            _history_stack=["older"],
            _grading_history_pending=False,
            _pending_params=object(),
        )

        MainWindow._rollback_pending_grading(fake_window)

        self.assertEqual(fake_window._history_stack, ["older"])
        self.assertFalse(fake_window._grading_history_pending)
        self.assertIsNone(fake_window._pending_params)

    def test_grading_success_clears_pending_history_marker(self):
        from src.ui.main_window import MainWindow

        params = object()
        calls = []
        fake_window = SimpleNamespace(
            progress_bar=FakeProgressBar(),
            current_image=None,
            _pending_params=params,
            _current_params=None,
            _grading_history_pending=True,
            update_image_display=lambda: calls.append("display"),
            _update_action_states=lambda: calls.append("state"),
        )
        result = np.full((2, 2, 3), 127, dtype=np.uint8)

        MainWindow._on_grading_finished(fake_window, result)

        self.assertTrue(fake_window.progress_bar.hidden)
        self.assertIs(fake_window.current_image, result)
        self.assertIs(fake_window._current_params, params)
        self.assertIsNone(fake_window._pending_params)
        self.assertFalse(fake_window._grading_history_pending)
        self.assertEqual(calls, ["display", "state"])

    def test_image_load_failure_updates_status_feedback(self):
        from src.ui.main_window import MainWindow

        fake_window = SimpleNamespace(statusbar=FakeStatusBar())

        with mock.patch("src.ui.main_window.imread_safe", return_value=None), \
                mock.patch("src.ui.main_window.QMessageBox.critical") as critical:
            MainWindow.load_image(fake_window, "broken.png")

        critical.assert_called_once()
        self.assertEqual(
            fake_window.statusbar.messages[-1],
            ("加载图像失败: broken.png", 5000),
        )

    def test_library_image_load_failure_is_visible(self):
        from src.ui.main_window import MainWindow

        fake_window = SimpleNamespace(statusbar=FakeStatusBar())

        with mock.patch("src.ui.main_window.imread_safe", return_value=None), \
                mock.patch("src.ui.main_window.QMessageBox.warning") as warning:
            MainWindow.load_reference_image(fake_window, "missing.png")

        warning.assert_called_once()
        self.assertEqual(
            fake_window.statusbar.messages[-1],
            ("无法加载图片: missing.png", 3000),
        )

    def test_text_reference_failure_restores_busy_state(self):
        from src.ui.main_window import MainWindow

        parser = FakeParser()
        progress_bar = FakeProgressBar()
        color_panel = FakeColorPanel()
        fake_window = SimpleNamespace(
            original_image=np.zeros((2, 2, 3), dtype=np.uint8),
            progress_bar=progress_bar,
            statusbar=FakeStatusBar(),
            color_panel=color_panel,
            nlp_parser=parser,
            color_engine=object(),
            image_db=FakeLibraryDb([], fail_search=True),
            _ensure_model=lambda _name: True,
        )
        fake_window._set_text_analysis_busy = (
            lambda busy: MainWindow._set_text_analysis_busy(fake_window, busy)
        )
        fake_window._resolve_text_reference_params = (
            lambda text: MainWindow._resolve_text_reference_params(fake_window, text)
        )

        with mock.patch("src.ui.main_window.QMessageBox.warning") as warning:
            MainWindow.process_text_command(fake_window, "参考这张图调成胶片感")

        warning.assert_called_once()
        self.assertEqual(parser.calls, [])
        self.assertTrue(progress_bar.hidden)
        self.assertFalse(progress_bar.shown)
        self.assertEqual(progress_bar.ranges, [(0, 0)])
        self.assertTrue(color_panel.apply_btn.enabled)
        self.assertEqual(color_panel.apply_btn.text, "应用")
        self.assertEqual(
            fake_window.statusbar.messages[-1],
            ("参考图片分析失败: index offline", 5000),
        )


if __name__ == "__main__":
    unittest.main()
