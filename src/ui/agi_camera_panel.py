"""
AGI相机面板
3D生成和动画预览
支持物体选择
"""
import numpy as np
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
    QFrame, QProgressBar, QFileDialog, QCheckBox, QSizePolicy,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint, QEvent
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QColor, QPen
import cv2

import sys
from pathlib import Path
# 使用相对导入
from ..ai import Mesh3D
from .ui_utils import WheelBlocker
from .image_picker_dialog import pick_images



class ClickableImageLabel(QWidget):
    """可点击的图像控件，用于物体选择"""

    clicked = Signal(int, int)  # 发送点击坐标
    box_selected = Signal(int, int, int, int)  # 发送框选区域
    path_selected = Signal(list)  # 发送划线路径 (list of points)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 350)  # 增大最小尺寸以便精准选择
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #404040;")

        # 启用鼠标追踪
        self.setMouseTracking(True)

        self._image = None
        self._pixmap = None
        self._display_scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._selection_mode = "point"  # "point", "box", or "path"
        self._box_start = None
        self._box_end = None
        self._drawing_box = False

        # 划线路径相关
        self._path_points = []  # 划线的点集合 (屏幕坐标)
        self._path_img_points = []  # 划线的点集合 (图像坐标)
        self._drawing_path = False

        # 事件处理状态变量（用于统一的拖拽/框选检测）
        self._press_pos = None  # 按下时的屏幕坐标
        self._press_img_pos = None  # 按下时的图像坐标
        self._is_dragging = False  # 是否正在拖动
        self._drag_threshold = 10  # 拖动阈值（像素），超过此距离才认为是框选

    def set_image(self, image: np.ndarray):
        """设置图像"""
        self._image = image.copy()
        self._update_display()

    def set_selection_mode(self, mode: str):
        """设置选择模式 ('point', 'box', 或 'path')"""
        self._selection_mode = mode
        self._box_start = None
        self._box_end = None
        self._drawing_box = False
        self._path_points = []
        self._path_img_points = []
        self._drawing_path = False

    def _update_display(self):
        """更新显示"""
        if self._image is None:
            self._pixmap = None
            self.update()
            return

        h, w = self._image.shape[:2]
        max_w = self.width() - 10
        max_h = self.height() - 10

        if max_w <= 0 or max_h <= 0:
            return

        scale = min(max_w / w, max_h / h)
        self._display_scale = scale

        new_w = int(w * scale)
        new_h = int(h * scale)

        self._offset_x = (self.width() - new_w) // 2
        self._offset_y = (self.height() - new_h) // 2

        display_image = cv2.resize(self._image, (new_w, new_h))

        # 如果正在绘制框选
        if self._drawing_box and self._box_start and self._box_end:
            x1 = int(self._box_start[0] * scale)
            y1 = int(self._box_start[1] * scale)
            x2 = int(self._box_end[0] * scale)
            y2 = int(self._box_end[1] * scale)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 如果正在绘制划线路径
        if self._drawing_path and len(self._path_img_points) > 1:
            # 将图像坐标的路径点转换为显示坐标
            scaled_points = [(int(p[0] * scale), int(p[1] * scale)) for p in self._path_img_points]
            # 绘制路径
            for i in range(len(scaled_points) - 1):
                cv2.line(display_image, scaled_points[i], scaled_points[i+1], (0, 255, 255), 3)
            # 绘制点
            for pt in scaled_points:
                cv2.circle(display_image, pt, 3, (255, 0, 0), -1)

        rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img_data = np.ascontiguousarray(rgb).copy()
        q_image = QImage(img_data.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image.copy())
        self.update()

    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        if self._pixmap:
            painter.drawPixmap(self._offset_x, self._offset_y, self._pixmap)
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先加载图像")

    def _screen_to_image_coords(self, x: int, y: int) -> tuple:
        """将屏幕坐标转换为图像坐标"""
        if self._image is None:
            return (0, 0)

        img_x = int((x - self._offset_x) / self._display_scale)
        img_y = int((y - self._offset_y) / self._display_scale)

        h, w = self._image.shape[:2]
        img_x = max(0, min(w - 1, img_x))
        img_y = max(0, min(h - 1, img_y))

        return (img_x, img_y)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            if self._image is None:
                print("[ClickableImageLabel] 图像未加载，忽略点击")
                return

            screen_x, screen_y = event.pos().x(), event.pos().y()
            img_x, img_y = self._screen_to_image_coords(screen_x, screen_y)

            # 记录按下位置
            self._press_pos = (screen_x, screen_y)
            self._press_img_pos = (img_x, img_y)
            self._is_dragging = False

            print(f"[ClickableImageLabel] 鼠标按下: 屏幕({screen_x}, {screen_y}), 图像({img_x}, {img_y}), 模式: {self._selection_mode}")

            # 如果是框选模式，准备框选
            if self._selection_mode == "box":
                self._box_start = (img_x, img_y)
                self._box_end = None
                self._drawing_box = True

            # 如果是划线模式，开始划线
            elif self._selection_mode == "path":
                self._path_points = [(screen_x, screen_y)]
                self._path_img_points = [(img_x, img_y)]
                self._drawing_path = True

            # 接受事件
            event.accept()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self._press_pos is None or self._image is None:
            return

        screen_x, screen_y = event.pos().x(), event.pos().y()

        # 框选模式：更新框选区域
        if self._selection_mode == "box" and self._drawing_box:
            img_x, img_y = self._screen_to_image_coords(screen_x, screen_y)
            self._box_end = (img_x, img_y)
            self._is_dragging = True
            self._update_display()
            event.accept()

        # 划线模式：记录路径点
        elif self._selection_mode == "path" and self._drawing_path:
            img_x, img_y = self._screen_to_image_coords(screen_x, screen_y)
            # 避免点太密集，保持一定间距
            if len(self._path_points) == 0 or \
               (abs(screen_x - self._path_points[-1][0]) > 3 or abs(screen_y - self._path_points[-1][1]) > 3):
                self._path_points.append((screen_x, screen_y))
                self._path_img_points.append((img_x, img_y))
                self._is_dragging = True
                self._update_display()
            event.accept()

        # 点选模式：检测是否有足够拖动距离（用于自动切换到框选）
        elif self._selection_mode == "point":
            dx = abs(screen_x - self._press_pos[0])
            dy = abs(screen_y - self._press_pos[1])
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > self._drag_threshold:
                self._is_dragging = True

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return

        screen_x, screen_y = event.pos().x(), event.pos().y()
        img_x, img_y = self._screen_to_image_coords(screen_x, screen_y)

        print(f"[ClickableImageLabel] 鼠标释放: 模式={self._selection_mode}, 拖动={self._is_dragging}")

        if self._selection_mode == "box" and self._drawing_box:
            # 框选模式：发送框选信号
            self._box_end = (img_x, img_y)
            self._drawing_box = False

            if self._box_start and self._box_end:
                x1, y1 = self._box_start
                x2, y2 = self._box_end
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # 只有当框选区域足够大时才发送信号
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    print(f"[ClickableImageLabel] 发送框选信号: ({x1}, {y1}) - ({x2}, {y2})")
                    self.box_selected.emit(x1, y1, x2, y2)
                else:
                    print("[ClickableImageLabel] 框选区域太小，忽略")

            self._update_display()

        elif self._selection_mode == "path" and self._drawing_path:
            # 划线模式：发送路径信号
            self._drawing_path = False

            if len(self._path_img_points) > 2:
                # 闭合路径（自动连接起点和终点）
                print(f"[ClickableImageLabel] 发送划线路径信号: {len(self._path_img_points)} 个点")
                self.path_selected.emit(self._path_img_points.copy())
            else:
                print("[ClickableImageLabel] 划线路径点太少，忽略")

            # 清除路径显示
            self._path_points = []
            self._path_img_points = []
            self._update_display()

        else:
            # 点选模式：发送点选信号
            if self._press_img_pos and not self._is_dragging:
                x, y = self._press_img_pos
                print(f"[ClickableImageLabel] 发送点选信号: ({x}, {y})")
                self.clicked.emit(x, y)

        # 重置状态
        self._press_pos = None
        self._press_img_pos = None
        self._is_dragging = False
        self._box_start = None
        self._box_end = None

        event.accept()

    def resizeEvent(self, event):
        """调整大小事件"""
        super().resizeEvent(event)
        self._update_display()


class AnimationPreview(QLabel):
    """动画预览组件"""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 350)  # 增大最小尺寸以便完整预览
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #404040;")

        self._frames: List[np.ndarray] = []
        self._current_frame = 0
        self._playing = False

        self._timer = QTimer()
        self._timer.timeout.connect(self._next_frame)

        self.setText("预览区域\n生成3D模型后可在此查看")

    def set_frames(self, frames: List[np.ndarray]):
        """设置动画帧"""
        self._frames = frames
        self._current_frame = 0
        if frames:
            self._show_frame(0)

    def _show_frame(self, index: int):
        """显示指定帧"""
        if not self._frames or index >= len(self._frames):
            return

        frame = self._frames[index]

        # 调整大小以适应显示区域
        h, w = frame.shape[:2]
        max_size = min(self.width() - 10, self.height() - 10)
        if max_size <= 0:
            max_size = 100
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

        # 转换为QPixmap - 确保数据独立拷贝避免悬垂指针
        if len(frame.shape) == 2:
            img_data = np.ascontiguousarray(frame).copy()
            q_image = QImage(img_data.data, new_w, new_h, new_w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_data = np.ascontiguousarray(rgb).copy()
            q_image = QImage(img_data.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)

        self.setPixmap(QPixmap.fromImage(q_image.copy()))

    def _next_frame(self):
        """切换到下一帧"""
        if not self._frames:
            return

        self._current_frame = (self._current_frame + 1) % len(self._frames)
        self._show_frame(self._current_frame)

    def play(self, fps: int = 30):
        """播放动画"""
        if self._frames:
            self._playing = True
            interval = int(1000 / fps)  # 根据fps计算间隔
            self._timer.start(interval)

    def pause(self):
        """暂停动画"""
        self._playing = False
        self._timer.stop()

    def stop(self):
        """停止动画"""
        self._playing = False
        self._timer.stop()
        self._current_frame = 0
        if self._frames:
            self._show_frame(0)

    def is_playing(self) -> bool:
        """是否正在播放"""
        return self._playing

    def set_frame_rate(self, fps: int):
        """设置帧率"""
        interval = int(1000 / fps)
        if self._playing:
            self._timer.setInterval(interval)


class AGICameraPanel(QWidget):
    """AGI相机面板"""

    generate_3d_requested = Signal(dict)  # 参数字典
    generate_animation_requested = Signal(dict)
    object_selected = Signal(int, int)  # 点选物体
    object_box_selected = Signal(int, int, int, int)  # 框选物体
    object_path_selected = Signal(list)  # 划线选择物体 (路径点列表)
    generate_object_3d_requested = Signal(dict)  # 从选中物体生成3D
    generate_multiview_3d_requested = Signal(dict)  # 多视角3D重建
    export_3d_model_requested = Signal()  # 导出3D模型

    def __init__(self):
        super().__init__()
        self._mesh: Optional[Mesh3D] = None
        self._frames: List[np.ndarray] = []
        self._wheel_blocker = WheelBlocker(self)
        self._has_selection = False
        self._pending_image = None
        self._multiview_images: List[str] = []  # 多视角图片路径列表
        self.image_db = None  # 图像数据库引用
        self._setup_ui()
        self._connect_signals()

        # 注意:不再使用全局事件过滤器，所有鼠标事件由 ClickableImageLabel 直接处理

    def set_database(self, image_db):
        """设置图像数据库"""
        self.image_db = image_db

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 物体选择区域
        selection_group = QGroupBox("物体选择 (点击或框选)")
        selection_layout = QVBoxLayout(selection_group)

        # 选择模式 - 放在图像上方
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("选择模式:"))

        self.point_mode_btn = QPushButton("点选")
        self.point_mode_btn.setCheckable(True)
        self.point_mode_btn.setChecked(True)
        self.point_mode_btn.setMinimumWidth(50)
        mode_layout.addWidget(self.point_mode_btn)

        self.box_mode_btn = QPushButton("框选")
        self.box_mode_btn.setCheckable(True)
        self.box_mode_btn.setMinimumWidth(50)
        mode_layout.addWidget(self.box_mode_btn)

        self.path_mode_btn = QPushButton("划线")
        self.path_mode_btn.setCheckable(True)
        self.path_mode_btn.setMinimumWidth(50)
        mode_layout.addWidget(self.path_mode_btn)

        mode_layout.addStretch()

        self.clear_selection_btn = QPushButton("清除选择")
        self.clear_selection_btn.setMinimumWidth(70)
        mode_layout.addWidget(self.clear_selection_btn)

        selection_layout.addLayout(mode_layout)

        # 图像视图 - 放在按钮下方
        self.image_view = ClickableImageLabel()
        selection_layout.addWidget(self.image_view)

        # 选择状态
        self.selection_status = QLabel("提示: 点击图像中的物体进行选择")
        self.selection_status.setStyleSheet("color: #888;")
        selection_layout.addWidget(self.selection_status)

        layout.addWidget(selection_group)

        # 预览区域
        preview_group = QGroupBox("3D预览")
        preview_layout = QVBoxLayout(preview_group)

        self.preview = AnimationPreview()
        preview_layout.addWidget(self.preview)

        # 播放控制
        control_layout = QHBoxLayout()

        self.play_btn = QPushButton("播放")
        self.play_btn.setCheckable(True)
        control_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("停止")
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()

        # 帧率控制
        control_layout.addWidget(QLabel("帧率:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.installEventFilter(self._wheel_blocker)
        control_layout.addWidget(self.fps_spin)

        preview_layout.addLayout(control_layout)
        layout.addWidget(preview_group)

        # 3D生成参数
        params_group = QGroupBox("生成参数")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)  # 增加间距

        # 深度缩放
        depth_layout = QHBoxLayout()
        depth_label = QLabel("深度强度:")
        depth_label.setFixedWidth(80)  # 固定标签宽度避免重叠
        depth_layout.addWidget(depth_label)
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(10, 150)  # 扩大范围以支持更立体的效果
        self.depth_slider.setValue(70)  # 提高默认值
        self.depth_slider.installEventFilter(self._wheel_blocker)
        depth_layout.addWidget(self.depth_slider)
        self.depth_label = QLabel("0.70")
        self.depth_label.setFixedWidth(45)  # 固定数值宽度
        depth_layout.addWidget(self.depth_label)
        params_layout.addLayout(depth_layout)

        # 网格分辨率
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("网格分辨率:")
        resolution_label.setFixedWidth(80)
        resolution_layout.addWidget(resolution_label)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["128", "256", "512"])
        self.resolution_combo.setCurrentText("256")
        self.resolution_combo.installEventFilter(self._wheel_blocker)
        resolution_layout.addWidget(self.resolution_combo)
        resolution_layout.addStretch()
        params_layout.addLayout(resolution_layout)

        # 旋转轴
        axis_layout = QHBoxLayout()
        axis_label = QLabel("旋转轴:")
        axis_label.setFixedWidth(80)
        axis_layout.addWidget(axis_label)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Y轴 (水平旋转)", "X轴 (垂直旋转)", "Z轴 (滚动)"])
        self.axis_combo.installEventFilter(self._wheel_blocker)
        axis_layout.addWidget(self.axis_combo)
        params_layout.addLayout(axis_layout)

        # 帧数
        frames_layout = QHBoxLayout()
        frames_label = QLabel("动画帧数:")
        frames_label.setFixedWidth(80)
        frames_layout.addWidget(frames_label)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(30, 120)
        self.frames_spin.setValue(60)
        self.frames_spin.installEventFilter(self._wheel_blocker)
        frames_layout.addWidget(self.frames_spin)
        frames_layout.addStretch()
        params_layout.addLayout(frames_layout)

        layout.addWidget(params_group)

        # 多视角重建区域
        multiview_group = QGroupBox("多视角3D重建")
        multiview_layout = QVBoxLayout(multiview_group)

        # 说明文字
        multiview_hint = QLabel("上传同一物体的多角度照片，可获得更精确的3D模型")
        multiview_hint.setStyleSheet("color: #888; font-size: 11px;")
        multiview_hint.setWordWrap(True)
        multiview_layout.addWidget(multiview_hint)

        # 图片列表显示
        self.multiview_list_label = QLabel("已选择: 0 张图片")
        multiview_layout.addWidget(self.multiview_list_label)

        # 按钮行
        multiview_btn_layout = QHBoxLayout()

        self.add_views_btn = QPushButton("添加图片")
        self.add_views_btn.setMinimumWidth(80)
        multiview_btn_layout.addWidget(self.add_views_btn)

        self.clear_views_btn = QPushButton("清空")
        self.clear_views_btn.setMinimumWidth(60)
        multiview_btn_layout.addWidget(self.clear_views_btn)

        multiview_btn_layout.addStretch()
        multiview_layout.addLayout(multiview_btn_layout)

        # 多视角生成按钮
        self.generate_multiview_3d_btn = QPushButton("多视角3D重建")
        self.generate_multiview_3d_btn.setMinimumHeight(40)
        self.generate_multiview_3d_btn.setEnabled(False)
        self.generate_multiview_3d_btn.setStyleSheet("QPushButton:disabled { color: #666; }")
        multiview_layout.addWidget(self.generate_multiview_3d_btn)

        layout.addWidget(multiview_group)

        # 生成按钮
        btn_layout = QVBoxLayout()

        self.generate_3d_btn = QPushButton("生成整图3D模型")
        self.generate_3d_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.generate_3d_btn)

        self.generate_object_3d_btn = QPushButton("生成选中物体3D")
        self.generate_object_3d_btn.setMinimumHeight(40)
        self.generate_object_3d_btn.setEnabled(False)
        self.generate_object_3d_btn.setStyleSheet("QPushButton:disabled { color: #666; }")
        btn_layout.addWidget(self.generate_object_3d_btn)

        self.generate_anim_btn = QPushButton("生成旋转动画")
        self.generate_anim_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.generate_anim_btn)

        layout.addLayout(btn_layout)

        # 导出按钮
        export_group = QGroupBox("导出")
        export_layout = QHBoxLayout(export_group)

        self.export_model_btn = QPushButton("导出3D模型")
        export_layout.addWidget(self.export_model_btn)

        self.export_gif_btn = QPushButton("导出GIF")
        export_layout.addWidget(self.export_gif_btn)

        self.export_video_btn = QPushButton("导出视频")
        export_layout.addWidget(self.export_video_btn)

        layout.addWidget(export_group)

        layout.addStretch()

    # 注意：eventFilter 和 _handle_mouse_* 方法已移除
    # 所有鼠标事件由 ClickableImageLabel 直接处理

    def _connect_signals(self):
        """连接信号"""
        # 深度滑块
        self.depth_slider.valueChanged.connect(
            lambda v: self.depth_label.setText(f"{v/100:.2f}")
        )

        # 选择模式 - 使用 toggled 信号而非 clicked，确保正确响应
        self.point_mode_btn.toggled.connect(self._on_point_mode_toggled)
        self.box_mode_btn.toggled.connect(self._on_box_mode_toggled)
        self.path_mode_btn.toggled.connect(self._on_path_mode_toggled)
        self.clear_selection_btn.clicked.connect(self._on_clear_selection)

        # 图像点击
        self.image_view.clicked.connect(self._on_image_clicked)
        self.image_view.box_selected.connect(self._on_box_selected)
        self.image_view.path_selected.connect(self._on_path_selected)

        # 播放控制
        self.play_btn.clicked.connect(self._on_play_clicked)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.fps_spin.valueChanged.connect(self.preview.set_frame_rate)

        # 生成按钮
        self.generate_3d_btn.clicked.connect(self._on_generate_3d)
        self.generate_object_3d_btn.clicked.connect(self._on_generate_object_3d)
        self.generate_anim_btn.clicked.connect(self._on_generate_animation)

        # 多视角重建按钮
        self.add_views_btn.clicked.connect(self._on_add_views)
        self.clear_views_btn.clicked.connect(self._on_clear_views)
        self.generate_multiview_3d_btn.clicked.connect(self._on_generate_multiview_3d)

        # 导出按钮
        self.export_model_btn.clicked.connect(self._export_model)
        self.export_gif_btn.clicked.connect(self._export_gif)
        self.export_video_btn.clicked.connect(self._export_video)

    def _on_point_mode_toggled(self, checked: bool):
        """切换到点选模式"""
        if checked:
            self.box_mode_btn.blockSignals(True)
            self.box_mode_btn.setChecked(False)
            self.box_mode_btn.blockSignals(False)
            self.path_mode_btn.blockSignals(True)
            self.path_mode_btn.setChecked(False)
            self.path_mode_btn.blockSignals(False)
            self.image_view.set_selection_mode("point")
            print("切换到点选模式")

    def _on_box_mode_toggled(self, checked: bool):
        """切换到框选模式"""
        if checked:
            self.point_mode_btn.blockSignals(True)
            self.point_mode_btn.setChecked(False)
            self.point_mode_btn.blockSignals(False)
            self.path_mode_btn.blockSignals(True)
            self.path_mode_btn.setChecked(False)
            self.path_mode_btn.blockSignals(False)
            self.image_view.set_selection_mode("box")
            print("切换到框选模式")

    def _on_path_mode_toggled(self, checked: bool):
        """切换到划线模式"""
        if checked:
            self.point_mode_btn.blockSignals(True)
            self.point_mode_btn.setChecked(False)
            self.point_mode_btn.blockSignals(False)
            self.box_mode_btn.blockSignals(True)
            self.box_mode_btn.setChecked(False)
            self.box_mode_btn.blockSignals(False)
            self.image_view.set_selection_mode("path")
            print("切换到划线模式")

    def _on_clear_selection(self):
        """清除选择"""
        self._has_selection = False
        self.generate_object_3d_btn.setEnabled(False)
        self.selection_status.setText("提示: 点击图像中的物体进行选择")
        self.selection_status.setStyleSheet("color: #888;")

        # 恢复原始图像显示（清除选择高亮）
        if hasattr(self, '_pending_image') and self._pending_image is not None:
            self.image_view.set_image(self._pending_image)

    def _on_image_clicked(self, x: int, y: int):
        """图像被点击"""
        print(f"[AGICameraPanel] 图像点击信号发出: ({x}, {y})")
        self.object_selected.emit(x, y)

    def _on_box_selected(self, x1: int, y1: int, x2: int, y2: int):
        """框选区域"""
        print(f"[AGICameraPanel] 框选信号发出: ({x1}, {y1}) - ({x2}, {y2})")
        self.object_box_selected.emit(x1, y1, x2, y2)

    def _on_path_selected(self, path_points: list):
        """划线路径选择"""
        print(f"[AGICameraPanel] 划线路径信号发出: {len(path_points)} 个点")
        self.object_path_selected.emit(path_points)

    def _on_generate_object_3d(self):
        """从选中物体生成3D"""
        self.generate_object_3d_requested.emit(self._get_params())

    def set_selection_result(self, success: bool, message: str = ""):
        """设置选择结果"""
        self._has_selection = success
        self.generate_object_3d_btn.setEnabled(success)
        if success:
            self.selection_status.setText("已选中物体 - 可以生成3D")
            self.selection_status.setStyleSheet("color: #4CAF50;")
        else:
            self.selection_status.setText(message or "选择失败，请重试")
            self.selection_status.setStyleSheet("color: #f44336;")

    def update_selection_preview(self, image: np.ndarray):
        """更新选择预览图像"""
        self.image_view.set_image(image)

    def _on_play_clicked(self, checked: bool):
        """播放按钮点击"""
        if checked:
            # 使用当前设置的帧率播放
            self.preview.play(fps=self.fps_spin.value())
            self.play_btn.setText("暂停")
        else:
            self.preview.pause()
            self.play_btn.setText("播放")

    def _on_stop_clicked(self):
        """停止按钮点击"""
        self.preview.stop()
        self.play_btn.setChecked(False)
        self.play_btn.setText("播放")

    def _get_params(self) -> dict:
        """获取当前参数"""
        axis_map = {"Y轴 (水平旋转)": "y", "X轴 (垂直旋转)": "x", "Z轴 (滚动)": "z"}
        return {
            "depth_scale": self.depth_slider.value() / 100.0,
            "resolution": int(self.resolution_combo.currentText()),
            "axis": axis_map.get(self.axis_combo.currentText(), "y"),
            "frames": self.frames_spin.value()
        }

    def _on_generate_3d(self):
        """生成3D模型"""
        self.generate_3d_requested.emit(self._get_params())

    def _on_generate_animation(self):
        """生成动画"""
        self.generate_animation_requested.emit(self._get_params())

    def set_image(self, image: np.ndarray):
        """设置图像用于物体选择"""
        print(f"[AGICameraPanel] set_image 调用, 图像尺寸: {image.shape}")
        self._pending_image = image.copy()  # 保存待设置的图像
        self.image_view.set_image(image)
        print(f"[AGICameraPanel] image_view._image is None: {self.image_view._image is None}")
        print(f"[AGICameraPanel] image_view 尺寸: {self.image_view.width()}x{self.image_view.height()}")
        print(f"[AGICameraPanel] image_view 可见: {self.image_view.isVisible()}")
        self._on_clear_selection()

    def showEvent(self, event):
        """面板显示事件"""
        super().showEvent(event)
        print(f"[AGICameraPanel] showEvent - 面板现在可见")
        print(f"[AGICameraPanel] image_view 尺寸: {self.image_view.width()}x{self.image_view.height()}")
        print(f"[AGICameraPanel] image_view 可见: {self.image_view.isVisible()}")
        # 如果有待设置的图像，在显示后重新更新显示
        if hasattr(self, '_pending_image') and self._pending_image is not None:
            self.image_view._update_display()

    def set_mesh(self, mesh: Mesh3D):
        """设置3D网格"""
        self._mesh = mesh
        # 显示一个静态预览 (使用简单投影)
        if mesh is not None:
            self._show_mesh_preview(mesh)

    def _show_mesh_preview(self, mesh: Mesh3D):
        """显示网格预览"""
        # 简单的正交投影预览
        h, w = 300, 300
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = [40, 40, 40]

        vertices = mesh.vertices
        colors = (mesh.colors * 255).astype(np.uint8)

        # 投影
        scale = min(h, w) * 0.4
        cx, cy = w // 2, h // 2

        projected = vertices[:, :2] * scale + [cx, cy]
        projected = projected.astype(int)

        # 绘制点
        for i, (pt, color) in enumerate(zip(projected, colors)):
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame, tuple(pt), 1, color.tolist(), -1)

        self.preview.set_frames([frame])

    def set_animation(self, frames: List[np.ndarray]):
        """设置动画帧"""
        self._frames = frames
        self.preview.set_frames(frames)

    def get_current_mesh(self) -> Optional[Mesh3D]:
        """获取当前网格"""
        return self._mesh

    def _export_model(self):
        """导出3D模型"""
        if self._mesh is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "提示", "请先生成3D模型")
            return

        # 发射信号给主窗口处理
        self.export_3d_model_requested.emit()

    def _export_gif(self):
        """导出GIF"""
        if not self._frames:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出GIF", "", "GIF (*.gif)"
        )

        if file_path:
            try:
                import imageio
                rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self._frames]
                imageio.mimsave(file_path, rgb_frames, fps=self.fps_spin.value(), loop=0)
            except ImportError:
                pass

    def _export_video(self):
        """导出视频"""
        if not self._frames:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出视频", "", "MP4 (*.mp4);;AVI (*.avi)"
        )

        if file_path:
            h, w = self._frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, self.fps_spin.value(), (w, h))

            for frame in self._frames:
                out.write(frame)

            out.release()

    def _on_add_views(self):
        """添加多视角图片"""
        # 使用统一的图像选择对话框
        file_paths = pick_images(self.image_db, multi_select=True, parent=self)

        if file_paths:
            self._multiview_images.extend(file_paths)
            self._update_multiview_label()

    def _on_clear_views(self):
        """清空多视角图片"""
        self._multiview_images.clear()
        self._update_multiview_label()

    def _update_multiview_label(self):
        """更新多视角图片数量标签"""
        count = len(self._multiview_images)
        self.multiview_list_label.setText(f"已选择: {count} 张图片")
        # 至少需要2张图片才能进行多视角重建
        self.generate_multiview_3d_btn.setEnabled(count >= 2)

    def _on_generate_multiview_3d(self):
        """请求多视角3D重建"""
        if len(self._multiview_images) < 2:
            return
        params = self._get_params()
        params['image_paths'] = self._multiview_images.copy()
        self.generate_multiview_3d_requested.emit(params)

    def get_multiview_images(self) -> List[str]:
        """获取多视角图片路径列表"""
        return self._multiview_images.copy()
