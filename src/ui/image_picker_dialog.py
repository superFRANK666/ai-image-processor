"""
统一的图像选择对话框
支持从文件系统或图像库选择图片
"""
import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import cv2

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QGridLayout, QScrollArea, QLineEdit,
    QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage

# 类型检查时导入
if TYPE_CHECKING:
    from ..ai import ImageIndexDatabase


class SelectableImageThumbnail(QWidget):
    """可选择的图像缩略图"""

    clicked = Signal(str, bool)  # 路径, 是否选中

    def __init__(self, image_path: str, size: int = 120, name: str = ""):
        super().__init__()
        self.image_path = image_path
        self.img_name = name
        self._selected = False

        self.setFixedWidth(size + 10)
        self._setup_ui(size)

    def _setup_ui(self, size):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 图片容器
        self.img_label = QLabel()
        self.img_label.setFixedSize(size, size)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 4px;
            }
        """)
        self.img_label.setCursor(Qt.PointingHandCursor)
        self._load_thumbnail(size)
        layout.addWidget(self.img_label)

        # 文字标签
        self.name_label = QLabel(self.img_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: #cccccc; font-size: 11px;")

        # 省略长文本
        font_metrics = self.name_label.fontMetrics()
        elided_text = font_metrics.elidedText(self.img_name, Qt.ElideMiddle, size)
        self.name_label.setText(elided_text)
        self.name_label.setToolTip(self.img_name)

        layout.addWidget(self.name_label)

    def _load_thumbnail(self, size: int):
        try:
            image = imread_safe(self.image_path)
            if image is not None:
                # 保持比例缩放
                h, w = image.shape[:2]
                scale = size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_data = np.ascontiguousarray(rgb).copy()
                q_image = QImage(img_data.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
                self.img_label.setPixmap(QPixmap.fromImage(q_image.copy()))
            else:
                self.img_label.setText("Err")
        except Exception:
            self.img_label.setText("Err")

    def set_selected(self, selected: bool):
        """设置选中状态"""
        self._selected = selected
        if selected:
            self.img_label.setStyleSheet("QLabel { background: #404040; border: 2px solid #0078d4; border-radius: 4px; }")
            self.name_label.setStyleSheet("color: #0078d4; font-weight: bold; font-size: 11px;")
        else:
            self.img_label.setStyleSheet("QLabel { background: #2d2d2d; border: 2px solid #404040; border-radius: 4px; }")
            self.name_label.setStyleSheet("color: #cccccc; font-size: 11px;")

    def is_selected(self) -> bool:
        return self._selected

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected = not self._selected
            self.set_selected(self._selected)
            self.clicked.emit(self.image_path, self._selected)


class ImagePickerDialog(QDialog):
    """图像选择对话框 - 支持从文件系统或图像库选择"""

    def __init__(self, image_db: Optional["ImageIndexDatabase"] = None,
                 multi_select: bool = True, parent=None):
        super().__init__(parent)
        self.image_db = image_db
        self.multi_select = multi_select
        self._selected_paths: List[str] = []
        self._thumbnails: List[SelectableImageThumbnail] = []

        self.setWindowTitle("选择图片" if multi_select else "选择图片")
        self.resize(900, 600)

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)

        # 选项卡
        self.tab_widget = QTabWidget()

        # 文件系统选项卡
        fs_tab = QWidget()
        fs_layout = QVBoxLayout(fs_tab)

        fs_hint = QLabel("从文件系统选择图片")
        fs_hint.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        fs_layout.addWidget(fs_hint)

        self.fs_path_label = QLabel("未选择文件")
        self.fs_path_label.setWordWrap(True)
        self.fs_path_label.setStyleSheet("padding: 10px; background: #2d2d2d; border-radius: 4px;")
        fs_layout.addWidget(self.fs_path_label)

        fs_btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("浏览文件...")
        self.browse_btn.setMinimumHeight(40)
        self.browse_btn.clicked.connect(self._on_browse_files)
        fs_btn_layout.addWidget(self.browse_btn)
        fs_layout.addLayout(fs_btn_layout)

        fs_layout.addStretch()
        self.tab_widget.addTab(fs_tab, "文件系统")

        # 图像库选项卡
        lib_tab = QWidget()
        lib_layout = QVBoxLayout(lib_tab)

        # 搜索栏
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索图片...")
        self.search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("搜索")
        self.search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(self.search_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_library)
        search_layout.addWidget(self.refresh_btn)

        lib_layout.addLayout(search_layout)

        # 状态标签
        self.status_label = QLabel("图像库: 0 张图片")
        lib_layout.addWidget(self.status_label)

        # 缩略图滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.thumbnail_widget)
        lib_layout.addWidget(scroll_area)

        self.tab_widget.addTab(lib_tab, "图像库")

        layout.addWidget(self.tab_widget)

        # 已选择的文件列表
        selection_label = QLabel("已选择:")
        layout.addWidget(selection_label)

        self.selection_list = QLabel("无")
        self.selection_list.setWordWrap(True)
        self.selection_list.setStyleSheet("padding: 10px; background: #2d2d2d; border-radius: 4px; min-height: 60px;")
        layout.addWidget(self.selection_list)

        # 按钮
        btn_layout = QHBoxLayout()

        self.clear_btn = QPushButton("清空选择")
        self.clear_btn.clicked.connect(self._clear_selection)
        btn_layout.addWidget(self.clear_btn)

        btn_layout.addStretch()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.setMinimumWidth(100)
        self.ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setMinimumWidth(100)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # 初始加载图像库
        if self.image_db:
            self._refresh_library()

    def _on_browse_files(self):
        """浏览文件系统"""
        try:
            from ..core.config import SUPPORTED_FORMATS
            formats = " ".join([f"*{fmt}" for fmt in SUPPORTED_FORMATS])
        except ImportError:
            formats = "*.jpg *.png *.jpeg *.bmp *.tiff"

        if self.multi_select:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图片", "",
                f"图像文件 ({formats});;所有文件 (*.*)"
            )
            if files:
                self._selected_paths = files
                self._update_selection_display()
        else:
            file, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "",
                f"图像文件 ({formats});;所有文件 (*.*)"
            )
            if file:
                self._selected_paths = [file]
                self._update_selection_display()

    def _on_search(self):
        """搜索图像库"""
        if not self.image_db:
            return

        query = self.search_input.text().strip()
        if query:
            results = self.image_db.search_by_text(query, top_k=50)
        else:
            results = self.image_db.get_images_by_group("全部", limit=50)

        self._show_thumbnails(results)

    def _refresh_library(self):
        """刷新图像库"""
        if not self.image_db:
            self.status_label.setText("图像库未初始化")
            return

        try:
            count = self.image_db.get_image_count()
            self.status_label.setText(f"图像库: {count} 张图片")

            # 加载前50张
            images = self.image_db.get_images_by_group("全部", limit=50)
            self._show_thumbnails(images)
        except Exception as e:
            self.status_label.setText(f"加载失败: {e}")

    def _show_thumbnails(self, images: List[Dict[str, Any]]):
        """显示缩略图"""
        # 清除现有缩略图
        for thumb in self._thumbnails:
            thumb.deleteLater()
        self._thumbnails.clear()

        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not images:
            return

        # 计算每行数量
        panel_width = self.thumbnail_widget.width() - 30
        thumb_size = 120
        cols = max(1, panel_width // (thumb_size + 10))

        for i, img_data in enumerate(images):
            image_path = img_data.get("path", "")
            if not image_path or not Path(image_path).exists():
                continue

            name = img_data.get("metadata", {}).get("name", Path(image_path).name)
            thumb = SelectableImageThumbnail(image_path, thumb_size, name)
            thumb.clicked.connect(self._on_thumbnail_clicked)

            # 如果已选中，标记为选中
            if image_path in self._selected_paths:
                thumb.set_selected(True)

            row = i // cols
            col = i % cols
            self.thumbnail_layout.addWidget(thumb, row, col)
            self._thumbnails.append(thumb)

    def _on_thumbnail_clicked(self, path: str, selected: bool):
        """缩略图点击"""
        if self.multi_select:
            # 多选模式
            if selected and path not in self._selected_paths:
                self._selected_paths.append(path)
            elif not selected and path in self._selected_paths:
                self._selected_paths.remove(path)
        else:
            # 单选模式：取消其他选择
            for thumb in self._thumbnails:
                if thumb.image_path != path:
                    thumb.set_selected(False)

            if selected:
                self._selected_paths = [path]
            else:
                self._selected_paths = []

        self._update_selection_display()

    def _update_selection_display(self):
        """更新选择显示"""
        if not self._selected_paths:
            self.selection_list.setText("无")
            self.fs_path_label.setText("未选择文件")
        else:
            # 显示选中的文件名
            names = [Path(p).name for p in self._selected_paths]
            if len(names) > 5:
                display = ", ".join(names[:5]) + f" ... (共{len(names)}个文件)"
            else:
                display = ", ".join(names)

            self.selection_list.setText(display)
            self.fs_path_label.setText(display)

    def _clear_selection(self):
        """清空选择"""
        self._selected_paths = []
        for thumb in self._thumbnails:
            thumb.set_selected(False)
        self._update_selection_display()

    def get_selected_paths(self) -> List[str]:
        """获取选中的路径"""
        return self._selected_paths.copy()


def pick_images(image_db: Optional["ImageIndexDatabase"] = None,
                multi_select: bool = True,
                parent=None) -> Optional[List[str]]:
    """
    便捷函数：打开图像选择对话框

    Args:
        image_db: 图像数据库实例
        multi_select: 是否多选
        parent: 父窗口

    Returns:
        选中的文件路径列表，如果取消则返回None
    """
    dialog = ImagePickerDialog(image_db, multi_select, parent)
    if dialog.exec() == QDialog.Accepted:
        paths = dialog.get_selected_paths()
        return paths if paths else None
    return None
