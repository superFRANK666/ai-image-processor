"""
图像查看器组件
支持缩放、平移、对比视图
"""
from pathlib import Path
import numpy as np
import cv2
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSlider, QPushButton, QFrame, QStackedLayout
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent, QFont
from .ui_utils import WheelBlocker


class ImageLabel(QLabel):
    """可交互的图像标签"""

    zoom_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)

        self._image: Optional[np.ndarray] = None
        self._compare_image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None

        self._zoom = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0

        self._pan_start: Optional[QPoint] = None
        self._pan_offset = QPoint(0, 0)

        self._compare_mode = False
        self._compare_position = 0.5  # 对比分割位置

        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray):
        """设置显示图像"""
        self._image = image
        self._update_display()

    def set_compare_mode(self, original: Optional[np.ndarray],
                         processed: Optional[np.ndarray]):
        """设置对比模式"""
        if original is None or processed is None:
            self._compare_mode = False
            self._compare_image = None
        else:
            self._compare_mode = True
            self._compare_image = original
            self._image = processed
        self._update_display()

    def _update_display(self):
        """更新显示"""
        if self._image is None:
            self.clear()
            return

        if self._compare_mode and self._compare_image is not None:
            display_image = self._create_compare_image()
        else:
            display_image = self._image

        # 转换为QPixmap
        self._pixmap = self._numpy_to_pixmap(display_image)

        # 应用缩放
        scaled_size = self._pixmap.size() * self._zoom
        scaled_pixmap = self._pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def _create_compare_image(self) -> np.ndarray:
        """创建对比图像"""
        h, w = self._image.shape[:2]
        split_x = int(w * self._compare_position)

        result = self._image.copy()
        result[:, :split_x] = self._compare_image[:, :split_x]

        # 绘制分割线
        cv2.line(result, (split_x, 0), (split_x, h), (255, 255, 255), 2)

        return result

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """NumPy数组转QPixmap"""
        if len(image.shape) == 2:
            # 灰度图
            h, w = image.shape
            bytes_per_line = w
            # 使用 copy() 确保数据连续且拥有独立内存
            img_data = np.ascontiguousarray(image).copy()
            q_image = QImage(img_data.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # 彩色图 (BGR -> RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            # 使用 copy() 确保数据连续且拥有独立内存
            img_data = np.ascontiguousarray(rgb_image).copy()
            q_image = QImage(img_data.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 立即转换为 QPixmap 并返回副本，避免 QImage 引用已释放的内存
        return QPixmap.fromImage(q_image.copy())

    def zoom_in(self):
        """放大"""
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        """缩小"""
        self.set_zoom(self._zoom / 1.25)

    def set_zoom(self, zoom: float):
        """设置缩放比例"""
        self._zoom = max(self._min_zoom, min(self._max_zoom, zoom))
        self._update_display()
        self.zoom_changed.emit(self._zoom)

    def fit_to_view(self):
        """适应视图"""
        if self._pixmap is None:
            return

        # 计算适应窗口的缩放比例
        widget_size = self.size()
        pixmap_size = self._pixmap.size()

        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()

        self.set_zoom(min(scale_x, scale_y) * 0.95)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下"""
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton and self._compare_mode:
            # 在对比模式下拖动分割线
            self._update_compare_position(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动"""
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            self.move(self.pos() + delta)
        elif self._compare_mode and event.buttons() & Qt.LeftButton:
            self._update_compare_position(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放"""
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)

    def _update_compare_position(self, pos: QPoint):
        """更新对比分割位置"""
        if self._pixmap is None:
            return

        # 计算相对位置
        pixmap_rect = self.pixmap().rect()
        label_rect = self.rect()

        # 居中偏移
        offset_x = (label_rect.width() - pixmap_rect.width()) // 2

        relative_x = pos.x() - offset_x
        self._compare_position = max(0.0, min(1.0, relative_x / pixmap_rect.width()))
        self._update_display()


class ImageViewer(QWidget):
    """图像查看器"""

    open_requested = Signal()
    import_requested = Signal()
    image_dropped = Signal(str)
    compare_toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("imageViewer")
        self.setAcceptDrops(True)
        self._wheel_blocker = WheelBlocker(self)
        self._has_image = False
        self._compare_enabled = False
        self._asset_path: Optional[str] = None
        self._canvas_status = "等待素材"
        self._canvas_detail = "拖入图片，或从右侧工作台打开素材。"
        self._setup_ui()
        self.set_canvas_context()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        canvas = QFrame()
        canvas.setObjectName("imageCanvas")
        canvas_layout = QStackedLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setStackingMode(QStackedLayout.StackAll)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setObjectName("imageScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)

        # 图像标签
        self.image_label = ImageLabel()
        self.image_label.setObjectName("imageViewport")
        scroll_area.setWidget(self.image_label)

        canvas_layout.addWidget(scroll_area)

        self.overlay = QWidget()
        self.overlay.setObjectName("viewerOverlay")
        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setSpacing(10)

        self.hud = QFrame()
        self.hud.setObjectName("canvasHud")
        hud_layout = QHBoxLayout(self.hud)
        hud_layout.setContentsMargins(12, 9, 12, 9)
        hud_layout.setSpacing(12)

        hud_text = QVBoxLayout()
        hud_text.setContentsMargins(0, 0, 0, 0)
        hud_text.setSpacing(2)
        self.asset_name_label = QLabel("未加载素材")
        self.asset_name_label.setObjectName("canvasAssetName")
        self.asset_meta_label = QLabel("等待输入")
        self.asset_meta_label.setObjectName("canvasAssetMeta")
        hud_text.addWidget(self.asset_name_label)
        hud_text.addWidget(self.asset_meta_label)
        hud_layout.addLayout(hud_text, 1)

        self.status_badge = QLabel("等待素材")
        self.status_badge.setObjectName("canvasBadge")
        self.compare_badge = QLabel("对比关闭")
        self.compare_badge.setObjectName("canvasBadge")
        self.compare_badge.setProperty("tone", "muted")
        hud_layout.addWidget(self.status_badge)
        hud_layout.addWidget(self.compare_badge)
        overlay_layout.addWidget(self.hud)

        overlay_layout.addStretch(1)
        empty_row = QHBoxLayout()
        empty_row.setContentsMargins(0, 0, 0, 0)
        empty_row.addStretch(1)
        self.empty_state = QFrame()
        self.empty_state.setObjectName("viewerEmptyState")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(28, 26, 28, 26)
        empty_layout.setSpacing(12)

        empty_title = QLabel("开始一次智能影像会话")
        empty_title.setObjectName("viewerEmptyTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_title)

        empty_subtitle = QLabel("打开素材后，这里会显示检视画布、缩放控制、对比状态和处理进度。")
        empty_subtitle.setObjectName("viewerEmptySubtitle")
        empty_subtitle.setWordWrap(True)
        empty_subtitle.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_subtitle)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 4, 0, 0)
        actions.setSpacing(8)
        self.empty_open_btn = QPushButton("打开素材")
        self.empty_open_btn.setProperty("variant", "primary")
        self.empty_open_btn.clicked.connect(self.open_requested.emit)
        self.empty_import_btn = QPushButton("导入图库")
        self.empty_import_btn.setProperty("variant", "secondary")
        self.empty_import_btn.clicked.connect(self.import_requested.emit)
        actions.addWidget(self.empty_open_btn)
        actions.addWidget(self.empty_import_btn)
        empty_layout.addLayout(actions)

        drop_hint = QLabel("支持拖入 JPG / PNG / WebP / BMP / TIFF")
        drop_hint.setObjectName("viewerDropHint")
        drop_hint.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(drop_hint)

        empty_row.addWidget(self.empty_state)

        self.processing_panel = QFrame()
        self.processing_panel.setObjectName("viewerProcessingState")
        processing_layout = QVBoxLayout(self.processing_panel)
        processing_layout.setContentsMargins(24, 22, 24, 22)
        processing_layout.setSpacing(10)
        self.processing_title = QLabel("正在处理")
        self.processing_title.setObjectName("viewerProcessingTitle")
        self.processing_title.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.processing_title)
        self.processing_detail = QLabel("请稍候，智能处理正在运行。")
        self.processing_detail.setObjectName("viewerProcessingDetail")
        self.processing_detail.setWordWrap(True)
        self.processing_detail.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.processing_detail)
        self.processing_hint = QLabel("完成后画布会自动更新")
        self.processing_hint.setObjectName("viewerProcessingHint")
        self.processing_hint.setAlignment(Qt.AlignCenter)
        processing_layout.addWidget(self.processing_hint)
        empty_row.addWidget(self.processing_panel)
        empty_row.addStretch(1)
        overlay_layout.addLayout(empty_row)
        overlay_layout.addStretch(2)

        canvas_layout.addWidget(self.overlay)
        canvas_layout.setCurrentWidget(self.overlay)
        self.overlay.raise_()
        layout.addWidget(canvas)

        # 底部工具栏
        toolbar = QFrame()
        toolbar.setObjectName("viewerToolbar")
        toolbar.setMaximumHeight(40)

        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        # 缩放控制
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)

        self.zoom_out_btn = QPushButton("－")
        self.zoom_out_btn.setFont(font)
        self.zoom_out_btn.setMinimumSize(32, 32)
        self.zoom_out_btn.setToolTip("缩小")
        # 微调样式以确保符号居中
        self.zoom_out_btn.setProperty("variant", "secondary")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(self.zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_slider.installEventFilter(self._wheel_blocker)
        toolbar_layout.addWidget(self.zoom_slider)

        self.zoom_in_btn = QPushButton("＋")
        self.zoom_in_btn.setFont(font)
        self.zoom_in_btn.setMinimumSize(32, 32)
        self.zoom_in_btn.setToolTip("放大")
        self.zoom_in_btn.setProperty("variant", "secondary")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar_layout.addWidget(self.zoom_in_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_label.setMinimumWidth(50)
        toolbar_layout.addWidget(self.zoom_label)

        toolbar_layout.addStretch()

        # 适应窗口按钮
        self.fit_btn = QPushButton("适应窗口")
        self.fit_btn.setProperty("variant", "secondary")
        self.fit_btn.clicked.connect(self.fit_to_view)
        toolbar_layout.addWidget(self.fit_btn)

        # 实际大小按钮
        self.actual_btn = QPushButton("实际大小")
        self.actual_btn.setProperty("variant", "secondary")
        self.actual_btn.clicked.connect(self.actual_size)
        toolbar_layout.addWidget(self.actual_btn)

        # 对比按钮
        self.compare_btn = QPushButton("对比")
        self.compare_btn.setProperty("variant", "secondary")
        self.compare_btn.setCheckable(True)
        self.compare_btn.toggled.connect(self.compare_toggled.emit)
        toolbar_layout.addWidget(self.compare_btn)

        layout.addWidget(toolbar)

        # 连接信号
        self.image_label.zoom_changed.connect(self._on_zoom_changed)
        self._refresh_canvas_state()

    def set_image(self, image: np.ndarray):
        """设置图像"""
        self._has_image = True
        self.image_label.set_image(image)
        height, width = image.shape[:2]
        self.set_canvas_context(width=width, height=height)

    def set_compare_mode(self, original: Optional[np.ndarray],
                         processed: Optional[np.ndarray]):
        """设置对比模式"""
        self._compare_enabled = original is not None and processed is not None
        self.image_label.set_compare_mode(original, processed)
        self._refresh_canvas_state()

    def set_canvas_context(
            self,
            file_path: Optional[str] = None,
            width: int = 0,
            height: int = 0,
            history_count: int = 0,
            library_count: Optional[int] = None,
            status: Optional[str] = None,
            detail: Optional[str] = None):
        """更新主画布上方的检视元信息。"""
        if file_path is not None:
            self._asset_path = file_path
        if status is not None:
            self._canvas_status = status
        if detail is not None:
            self._canvas_detail = detail

        if self._asset_path:
            path = Path(self._asset_path)
            self.asset_name_label.setText(path.name)
            self.asset_name_label.setToolTip(str(path))
        else:
            self.asset_name_label.setText("未加载素材")
            self.asset_name_label.setToolTip("")

        if self._has_image and width and height:
            parts = [f"{width} x {height}px", f"历史 {history_count} 步"]
            if library_count is not None:
                parts.append(f"图库 {library_count} 张")
            self.asset_meta_label.setText(" · ".join(parts))
        else:
            self.asset_meta_label.setText(self._canvas_detail)

        self.status_badge.setText(self._canvas_status)
        tone = self._tone_for_status(self._canvas_status)
        self.status_badge.setProperty("tone", tone)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.processing_title.setText(self._processing_title_for_status(self._canvas_status))
        self.processing_detail.setText(self._canvas_detail)
        self._refresh_canvas_state()

    def clear_image(self):
        """清空画布并恢复空状态。"""
        self._has_image = False
        self._compare_enabled = False
        self._asset_path = None
        self.image_label.clear()
        self.set_canvas_context(status="等待素材", detail="拖入图片，或从右侧工作台打开素材。")

    def zoom_in(self):
        """放大"""
        self.image_label.zoom_in()

    def zoom_out(self):
        """缩小"""
        self.image_label.zoom_out()

    def fit_to_view(self):
        """适应窗口"""
        self.image_label.fit_to_view()

    def actual_size(self):
        """实际大小"""
        self.image_label.set_zoom(1.0)

    def _on_zoom_slider_changed(self, value: int):
        """缩放滑块改变"""
        zoom = value / 100.0
        self.image_label.set_zoom(zoom)

    def _on_zoom_changed(self, zoom: float):
        """缩放改变回调"""
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _refresh_canvas_state(self):
        is_busy = self._tone_for_status(self._canvas_status) == "busy"
        self.empty_state.setVisible(not self._has_image and not is_busy)
        self.processing_panel.setVisible(is_busy)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, self._has_image and not is_busy)
        self.compare_badge.setText("对比开启" if self._compare_enabled else "对比关闭")
        self.compare_badge.setProperty("tone", "active" if self._compare_enabled else "muted")
        self.compare_badge.style().unpolish(self.compare_badge)
        self.compare_badge.style().polish(self.compare_badge)
        for control in (
            self.zoom_out_btn,
            self.zoom_slider,
            self.zoom_in_btn,
            self.fit_btn,
            self.actual_btn,
            self.compare_btn,
        ):
            control.setEnabled(self._has_image)

    def _tone_for_status(self, status: str) -> str:
        if status in {"处理中", "加载中", "分析中"}:
            return "busy"
        if status in {"失败", "错误"}:
            return "danger"
        if status in {"已更新", "已保存", "就绪", "对比中"}:
            return "active"
        return "muted"

    def _processing_title_for_status(self, status: str) -> str:
        if status == "加载中":
            return "正在加载"
        if status == "分析中":
            return "正在分析"
        return "正在处理"

    def dragEnterEvent(self, event):
        """接受用户拖入的图片文件。"""
        if self._first_supported_drop_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = self._first_supported_drop_path(event.mimeData())
        if file_path is None:
            event.ignore()
            return
        self.image_dropped.emit(file_path)
        event.acceptProposedAction()

    def _first_supported_drop_path(self, mime_data) -> Optional[str]:
        if not mime_data.hasUrls():
            return None
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if Path(path).suffix.lower() in supported:
                return path
        return None
