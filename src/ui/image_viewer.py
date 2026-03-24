"""
图像查看器组件
支持缩放、平移、对比视图
"""
import numpy as np
import cv2
from typing import Optional, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSlider, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QObject, QEvent
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QWheelEvent, QMouseEvent, QFont
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

    def __init__(self):
        super().__init__()
        self._wheel_blocker = WheelBlocker(self)
        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        # 图像标签
        self.image_label = ImageLabel()
        scroll_area.setWidget(self.image_label)

        layout.addWidget(scroll_area)

        # 底部工具栏
        toolbar = QFrame()
        toolbar.setMaximumHeight(40)
        toolbar.setStyleSheet("QFrame { background: #2d2d2d; border-top: 1px solid #404040; }")

        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)

        # 缩放控制
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFont(font)
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.setToolTip("缩小")
        # 微调样式以确保符号居中
        zoom_out_btn.setStyleSheet("padding: 0; padding-bottom: 2px;")
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar_layout.addWidget(zoom_out_btn)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(150)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.zoom_slider.installEventFilter(self._wheel_blocker)
        toolbar_layout.addWidget(self.zoom_slider)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFont(font)
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.setToolTip("放大")
        zoom_in_btn.setStyleSheet("padding: 0; padding-bottom: 2px;")
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar_layout.addWidget(zoom_in_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        toolbar_layout.addWidget(self.zoom_label)

        toolbar_layout.addStretch()

        # 适应窗口按钮
        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(self.fit_to_view)
        toolbar_layout.addWidget(fit_btn)

        # 实际大小按钮
        actual_btn = QPushButton("实际大小")
        actual_btn.clicked.connect(self.actual_size)
        toolbar_layout.addWidget(actual_btn)

        layout.addWidget(toolbar)

        # 连接信号
        self.image_label.zoom_changed.connect(self._on_zoom_changed)

    def set_image(self, image: np.ndarray):
        """设置图像"""
        self.image_label.set_image(image)

    def set_compare_mode(self, original: Optional[np.ndarray],
                         processed: Optional[np.ndarray]):
        """设置对比模式"""
        self.image_label.set_compare_mode(original, processed)

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
