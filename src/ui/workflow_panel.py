"""
智能工作流面板
为商业化创作工作台提供当前会话、模型状态和下一步动作概览。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class WorkflowMetrics:
    """当前创作会话的轻量指标。"""

    file_path: Optional[str] = None
    width: int = 0
    height: int = 0
    history_count: int = 0
    library_count: Optional[int] = None
    has_mesh: bool = False
    has_animation: bool = False
    has_selection: bool = False


class WorkflowPanel(QWidget):
    """面向创作者的智能工作流概览。"""

    open_image_requested = Signal()
    import_images_requested = Signal()
    grade_requested = Signal()
    find_similar_requested = Signal()
    generate_3d_requested = Signal()
    save_requested = Signal()
    tab_requested = Signal(str)

    MODEL_LABELS = {
        "color_engine": "调色引擎",
        "nlp_parser": "语义解析",
        "image_db": "图像库",
        "style_analyzer": "风格分析",
        "agi_camera": "3D 生成",
    }

    STEP_LABELS = {
        "load": "载入素材",
        "grade": "智能调色",
        "reference": "风格检索",
        "generate": "3D/动画",
        "export": "保存导出",
    }

    def __init__(self):
        super().__init__()
        self.setObjectName("workflowPanel")
        self._model_states: Dict[str, str] = {
            name: "待加载" for name in self.MODEL_LABELS
        }
        self._metrics = WorkflowMetrics()
        self._activity_limit = 8
        self._setup_ui()
        self.update_metrics(self._metrics)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("智能工作台")
        header.setObjectName("workflowTitle")
        layout.addWidget(header)

        subtitle = QLabel("围绕当前素材组织调色、检索、3D 生成和导出。")
        subtitle.setObjectName("workflowSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.session_group = QGroupBox("当前会话")
        session_layout = QGridLayout(self.session_group)
        session_layout.setHorizontalSpacing(10)
        session_layout.setVerticalSpacing(8)
        self.file_value = QLabel("未加载素材")
        self.file_value.setWordWrap(True)
        self.size_value = QLabel("-")
        self.history_value = QLabel("0 步")
        self.library_value = QLabel("图像库加载中")
        self.asset_value = QLabel("无生成产物")
        self._add_metric_row(session_layout, 0, "文件", self.file_value)
        self._add_metric_row(session_layout, 1, "尺寸", self.size_value)
        self._add_metric_row(session_layout, 2, "历史", self.history_value)
        self._add_metric_row(session_layout, 3, "图库", self.library_value)
        self._add_metric_row(session_layout, 4, "产物", self.asset_value)
        layout.addWidget(self.session_group)

        action_group = QGroupBox("下一步动作")
        action_layout = QGridLayout(action_group)
        action_layout.setHorizontalSpacing(8)
        action_layout.setVerticalSpacing(8)

        self.open_btn = self._make_action_button("打开素材", self.open_image_requested.emit, "primary")
        self.import_btn = self._make_action_button("导入图库", self.import_images_requested.emit)
        self.grade_btn = self._make_action_button("智能调色", self.grade_requested.emit, "primary")
        self.similar_btn = self._make_action_button("找相似", self.find_similar_requested.emit)
        self.generate_btn = self._make_action_button("生成 3D", self.generate_3d_requested.emit)
        self.save_btn = self._make_action_button("保存结果", self.save_requested.emit, "secondary")

        for index, button in enumerate(
            (self.open_btn, self.import_btn, self.grade_btn, self.similar_btn, self.generate_btn, self.save_btn)
        ):
            action_layout.addWidget(button, index // 2, index % 2)
        layout.addWidget(action_group)

        self.steps_group = QGroupBox("工作流进度")
        steps_layout = QVBoxLayout(self.steps_group)
        steps_layout.setSpacing(6)
        self.step_labels: Dict[str, QLabel] = {}
        for key, label in self.STEP_LABELS.items():
            step = QLabel(f"○ {label}")
            step.setObjectName("workflowStep")
            self.step_labels[key] = step
            steps_layout.addWidget(step)
        layout.addWidget(self.steps_group)

        models_group = QGroupBox("AI 能力状态")
        models_layout = QVBoxLayout(models_group)
        models_layout.setSpacing(6)
        self.model_labels: Dict[str, QLabel] = {}
        for name, label in self.MODEL_LABELS.items():
            row = QFrame()
            row.setObjectName("workflowModelRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name_label = QLabel(label)
            value_label = QLabel("待加载")
            value_label.setObjectName("workflowModelState")
            row_layout.addWidget(name_label)
            row_layout.addStretch()
            row_layout.addWidget(value_label)
            self.model_labels[name] = value_label
            models_layout.addWidget(row)
        layout.addWidget(models_group)

        activity_group = QGroupBox("最近活动")
        activity_layout = QVBoxLayout(activity_group)
        self.activity_list = QListWidget()
        self.activity_list.setMaximumHeight(150)
        activity_layout.addWidget(self.activity_list)
        layout.addWidget(activity_group)

        layout.addStretch()

    def _add_metric_row(self, layout: QGridLayout, row: int, name: str, value: QLabel):
        name_label = QLabel(name)
        name_label.setObjectName("workflowMetricName")
        value.setObjectName("workflowMetricValue")
        layout.addWidget(name_label, row, 0)
        layout.addWidget(value, row, 1)

    def _make_action_button(self, text: str, callback, variant: str = "secondary") -> QPushButton:
        button = QPushButton(text)
        button.setProperty("variant", variant)
        button.setMinimumHeight(34)
        button.clicked.connect(callback)
        return button

    def update_metrics(self, metrics: WorkflowMetrics):
        """更新当前创作会话指标。"""
        self._metrics = metrics
        has_image = bool(metrics.file_path)

        if metrics.file_path:
            path = Path(metrics.file_path)
            self.file_value.setText(path.name)
            self.file_value.setToolTip(str(path))
        else:
            self.file_value.setText("未加载素材")
            self.file_value.setToolTip("")

        self.size_value.setText(f"{metrics.width} x {metrics.height}" if has_image else "-")
        self.history_value.setText(f"{metrics.history_count} 步")
        if metrics.library_count is None:
            self.library_value.setText("图像库加载中")
        else:
            self.library_value.setText(f"{metrics.library_count} 张")

        assets = []
        if metrics.has_selection:
            assets.append("已选物体")
        if metrics.has_mesh:
            assets.append("3D 模型")
        if metrics.has_animation:
            assets.append("动画")
        self.asset_value.setText("、".join(assets) if assets else "无生成产物")

        self.grade_btn.setEnabled(has_image)
        self.similar_btn.setEnabled(has_image)
        self.generate_btn.setEnabled(has_image)
        self.save_btn.setEnabled(has_image)

        self._update_steps(metrics)

    def update_model_state(self, model_name: str, state: str):
        """更新单个 AI 模型状态。"""
        if model_name not in self.model_labels:
            return
        normalized = self._normalize_state(state)
        self._model_states[model_name] = normalized
        self.model_labels[model_name].setText(normalized)
        self.model_labels[model_name].setProperty("state", normalized)
        self.model_labels[model_name].style().unpolish(self.model_labels[model_name])
        self.model_labels[model_name].style().polish(self.model_labels[model_name])

    def sync_model_states(self, loaded: Iterable[str], loading: Iterable[str] = ()):
        """根据 ModelManager 快照同步模型状态。"""
        loaded_set = set(loaded)
        loading_set = set(loading)
        for model_name in self.MODEL_LABELS:
            if model_name in loaded_set:
                self.update_model_state(model_name, "就绪")
            elif model_name in loading_set:
                self.update_model_state(model_name, "加载中")
            else:
                self.update_model_state(model_name, self._model_states.get(model_name, "待加载"))

    def add_activity(self, message: str):
        """记录一条用户可见的工作流活动。"""
        if not message:
            return
        item = QListWidgetItem(message)
        self.activity_list.insertItem(0, item)
        while self.activity_list.count() > self._activity_limit:
            self.activity_list.takeItem(self.activity_list.count() - 1)

    def _normalize_state(self, state: str) -> str:
        if state in {"ready", "loaded", "就绪"}:
            return "就绪"
        if state in {"loading", "加载中"}:
            return "加载中"
        if state in {"failed", "失败"}:
            return "失败"
        return "待加载"

    def _update_steps(self, metrics: WorkflowMetrics):
        complete = {
            "load": bool(metrics.file_path),
            "grade": metrics.history_count > 0,
            "reference": metrics.library_count is not None and metrics.library_count > 0,
            "generate": metrics.has_mesh or metrics.has_animation,
            "export": False,
        }
        for key, label in self.step_labels.items():
            prefix = "●" if complete[key] else "○"
            label.setText(f"{prefix} {self.STEP_LABELS[key]}")
            label.setProperty("complete", complete[key])
            label.style().unpolish(label)
            label.style().polish(label)
