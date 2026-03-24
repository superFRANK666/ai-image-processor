"""
调色面板
支持自然语言输入和参数手动调整
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QGroupBox, QScrollArea, QFrame,
    QComboBox, QSpinBox, QDoubleSpinBox, QGridLayout
)
from PySide6.QtCore import Qt, Signal

import sys
from pathlib import Path
# 使用相对导入
from ..ai import ColorGradingParams
from PySide6.QtCore import QObject, QEvent
from .ui_utils import WheelBlocker



class ParamSlider(QWidget):
    """参数滑块组件"""

    value_changed = Signal(float)

    def __init__(self, name: str, min_val: float, max_val: float,
                 default: float = 0, decimals: int = 2, suffix: str = ""):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.scale = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)  # 增加间距避免元素重叠

        # 标签
        self.label = QLabel(name)
        self.label.setMinimumWidth(80)  # 增大最小宽度以避免文字被截断
        self.label.setFixedWidth(80)  # 固定宽度确保对齐
        layout.addWidget(self.label)

        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.scale), int(max_val * self.scale))
        self.slider.setValue(int(default * self.scale))
        self.slider.valueChanged.connect(self._on_slider_changed)
        
        # 禁用滚轮
        self._wheel_blocker = WheelBlocker(self)
        self.slider.installEventFilter(self._wheel_blocker)
        
        layout.addWidget(self.slider)

        # 数值显示
        self.value_label = QLabel(f"{default:.{decimals}f}{suffix}")
        self.value_label.setMinimumWidth(60)  # 增大最小宽度以避免数值被截断
        self.value_label.setFixedWidth(60)  # 固定宽度确保对齐
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.value_label)

        self.suffix = suffix

    def _on_slider_changed(self, value: int):
        """滑块值改变"""
        real_value = value / self.scale
        self.value_label.setText(f"{real_value:.{self.decimals}f}{self.suffix}")
        self.value_changed.emit(real_value)

    def get_value(self) -> float:
        """获取当前值"""
        return self.slider.value() / self.scale

    def set_value(self, value: float):
        """设置值"""
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self.scale))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{value:.{self.decimals}f}{self.suffix}")

    def reset(self):
        """重置到默认值"""
        self.slider.setValue(int((self.min_val + self.max_val) / 2 * self.scale))


class ColorGradingPanel(QWidget):
    """调色面板"""

    params_changed = Signal(object)  # ColorGradingParams
    text_input_submitted = Signal(str)
    find_similar_requested = Signal()
    upload_reference_requested = Signal()  # 请求上传参考图片

    def __init__(self):
        super().__init__()
        
        # 初始化滚轮屏蔽器 (必须在 setup_ui 之前)
        self._wheel_blocker = WheelBlocker(self)
        
        # 防抖定时器 - 优化滑块调节性能
        from PySide6.QtCore import QTimer
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)  # 150ms 防抖延迟
        self._debounce_timer.timeout.connect(self._emit_params_changed)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 自然语言输入区域
        nlp_group = QGroupBox("一句话调色")
        nlp_layout = QVBoxLayout(nlp_group)

        # 输入框
        input_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("例如: 复刻去年海边旅行照片的蓝调色调")
        input_layout.addWidget(self.text_input)

        self.apply_btn = QPushButton("应用")
        self.apply_btn.setMinimumWidth(60)
        input_layout.addWidget(self.apply_btn)
        nlp_layout.addLayout(input_layout)

        # 快捷预设
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("快捷预设:"))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "选择预设...", "蓝调", "暖调", "复古", "电影感",
            "日系", "黑金", "清新", "梦幻", "赛博朋克"
        ])
        self.preset_combo.installEventFilter(self._wheel_blocker)
        preset_layout.addWidget(self.preset_combo)
        nlp_layout.addLayout(preset_layout)

        # 风格参考功能区
        style_ref_layout = QHBoxLayout()
        
        # 查找相似按钮
        self.find_similar_btn = QPushButton("查找相似风格")
        self.find_similar_btn.setToolTip("在图像库中查找与当前图片风格相似的图片")
        style_ref_layout.addWidget(self.find_similar_btn)
        
        # 上传参考图片按钮
        self.upload_reference_btn = QPushButton("上传参考图片")
        self.upload_reference_btn.setToolTip("上传一张参考图片，提取其色调并应用到当前图片")
        style_ref_layout.addWidget(self.upload_reference_btn)
        
        nlp_layout.addLayout(style_ref_layout)
        
        # 参考图片状态标签
        self.reference_status_label = QLabel("")
        self.reference_status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.reference_status_label.setWordWrap(True)
        nlp_layout.addWidget(self.reference_status_label)

        layout.addWidget(nlp_group)

        # 参数调整区域 (可滚动)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setSpacing(5)

        # 基础调整
        basic_group = QGroupBox("基础调整")
        basic_layout = QVBoxLayout(basic_group)

        self.exposure_slider = ParamSlider("曝光", -2.0, 2.0, 0.0, 2)
        basic_layout.addWidget(self.exposure_slider)

        self.contrast_slider = ParamSlider("对比度", 0.5, 2.0, 1.0, 2)
        basic_layout.addWidget(self.contrast_slider)

        self.highlights_slider = ParamSlider("高光", -100, 100, 0, 0)
        basic_layout.addWidget(self.highlights_slider)

        self.shadows_slider = ParamSlider("阴影", -100, 100, 0, 0)
        basic_layout.addWidget(self.shadows_slider)

        self.whites_slider = ParamSlider("白色", -100, 100, 0, 0)
        basic_layout.addWidget(self.whites_slider)

        self.blacks_slider = ParamSlider("黑色", -100, 100, 0, 0)
        basic_layout.addWidget(self.blacks_slider)

        params_layout.addWidget(basic_group)

        # 颜色调整
        color_group = QGroupBox("颜色调整")
        color_layout = QVBoxLayout(color_group)

        self.temperature_slider = ParamSlider("色温", -100, 100, 0, 0)
        color_layout.addWidget(self.temperature_slider)

        self.tint_slider = ParamSlider("色调", -100, 100, 0, 0)
        color_layout.addWidget(self.tint_slider)

        self.vibrance_slider = ParamSlider("自然饱和度", -100, 100, 0, 0)
        color_layout.addWidget(self.vibrance_slider)

        self.saturation_slider = ParamSlider("饱和度", 0.0, 2.0, 1.0, 2)
        color_layout.addWidget(self.saturation_slider)

        self.hue_slider = ParamSlider("色相偏移", -180, 180, 0, 0, "°")
        color_layout.addWidget(self.hue_slider)

        params_layout.addWidget(color_group)

        # 效果
        effects_group = QGroupBox("效果")
        effects_layout = QVBoxLayout(effects_group)

        self.clarity_slider = ParamSlider("清晰度", -100, 100, 0, 0)
        effects_layout.addWidget(self.clarity_slider)

        self.dehaze_slider = ParamSlider("去雾", -100, 100, 0, 0)
        effects_layout.addWidget(self.dehaze_slider)

        self.vignette_slider = ParamSlider("暗角", 0, 100, 0, 0)
        effects_layout.addWidget(self.vignette_slider)

        self.grain_slider = ParamSlider("颗粒", 0, 100, 0, 0)
        effects_layout.addWidget(self.grain_slider)

        self.fade_slider = ParamSlider("褪色", 0, 1.0, 0, 2)
        effects_layout.addWidget(self.fade_slider)

        params_layout.addWidget(effects_group)

        params_layout.addStretch()
        scroll_area.setWidget(params_widget)
        layout.addWidget(scroll_area)

        # 底部按钮
        btn_layout = QHBoxLayout()

        self.reset_btn = QPushButton("重置参数")
        btn_layout.addWidget(self.reset_btn)

        self.copy_params_btn = QPushButton("复制参数")
        btn_layout.addWidget(self.copy_params_btn)

        layout.addLayout(btn_layout)

    def _connect_signals(self):
        """连接信号"""
        # 文本输入
        self.text_input.returnPressed.connect(self._on_text_submitted)
        self.apply_btn.clicked.connect(self._on_text_submitted)

        # 预设选择
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)

        # 查找相似
        self.find_similar_btn.clicked.connect(self.find_similar_requested.emit)
        
        # 上传参考图片
        self.upload_reference_btn.clicked.connect(self.upload_reference_requested.emit)

        # 滑块变化
        self.exposure_slider.value_changed.connect(self._on_param_changed)
        self.contrast_slider.value_changed.connect(self._on_param_changed)
        self.highlights_slider.value_changed.connect(self._on_param_changed)
        self.shadows_slider.value_changed.connect(self._on_param_changed)
        self.whites_slider.value_changed.connect(self._on_param_changed)
        self.blacks_slider.value_changed.connect(self._on_param_changed)
        self.temperature_slider.value_changed.connect(self._on_param_changed)
        self.tint_slider.value_changed.connect(self._on_param_changed)
        self.vibrance_slider.value_changed.connect(self._on_param_changed)
        self.saturation_slider.value_changed.connect(self._on_param_changed)
        self.hue_slider.value_changed.connect(self._on_param_changed)
        self.clarity_slider.value_changed.connect(self._on_param_changed)
        self.dehaze_slider.value_changed.connect(self._on_param_changed)
        self.vignette_slider.value_changed.connect(self._on_param_changed)
        self.grain_slider.value_changed.connect(self._on_param_changed)
        self.fade_slider.value_changed.connect(self._on_param_changed)

        # 重置按钮
        self.reset_btn.clicked.connect(self.reset_params)

        # 复制参数
        self.copy_params_btn.clicked.connect(self._copy_params)

    def _on_text_submitted(self):
        """文本提交"""
        text = self.text_input.text().strip()
        if text:
            self.text_input_submitted.emit(text)

    def _on_preset_selected(self, preset: str):
        """预设选择"""
        if preset and preset != "选择预设...":
            # 设置输入框文本
            self.text_input.setText(preset)
            
            # 临时改变按钮文字显示正在处理
            original_text = self.apply_btn.text()
            self.apply_btn.setText("处理中...")
            self.apply_btn.setEnabled(False)
            
            # 发送信号
            self.text_input_submitted.emit(preset)
            
            # 恢复按钮状态
            from PySide6.QtCore import QTimer
            QTimer.singleShot(500, lambda: self._restore_apply_button(original_text))
            
            # 重置下拉框到默认选项，允许再次选择同一预设
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self.preset_combo.blockSignals(False)
    
    def _restore_apply_button(self, text: str):
        """恢复应用按钮状态"""
        self.apply_btn.setText(text)
        self.apply_btn.setEnabled(True)

    def _on_param_changed(self, value: float):
        """参数改变 - 使用防抖机制"""
        # 重启防抖定时器，只有在用户停止调节后才触发处理
        self._debounce_timer.start()
    
    def _emit_params_changed(self):
        """实际发送参数变化信号"""
        params = self.get_params()
        self.params_changed.emit(params)

    def get_params(self) -> ColorGradingParams:
        """获取当前参数"""
        return ColorGradingParams(
            exposure=self.exposure_slider.get_value(),
            contrast=self.contrast_slider.get_value(),
            highlights=self.highlights_slider.get_value(),
            shadows=self.shadows_slider.get_value(),
            whites=self.whites_slider.get_value(),
            blacks=self.blacks_slider.get_value(),
            temperature=self.temperature_slider.get_value(),
            tint=self.tint_slider.get_value(),
            vibrance=self.vibrance_slider.get_value(),
            saturation=self.saturation_slider.get_value(),
            hue_shift=self.hue_slider.get_value(),
            clarity=self.clarity_slider.get_value(),
            dehaze=self.dehaze_slider.get_value(),
            vignette=self.vignette_slider.get_value(),
            grain=self.grain_slider.get_value(),
            fade=self.fade_slider.get_value(),
        )

    def set_params(self, params: ColorGradingParams):
        """设置参数"""
        # 阻止信号发送以避免循环
        self.blockSignals(True)

        self.exposure_slider.set_value(params.exposure)
        self.contrast_slider.set_value(params.contrast)
        self.highlights_slider.set_value(params.highlights)
        self.shadows_slider.set_value(params.shadows)
        self.whites_slider.set_value(params.whites)
        self.blacks_slider.set_value(params.blacks)
        self.temperature_slider.set_value(params.temperature)
        self.tint_slider.set_value(params.tint)
        self.vibrance_slider.set_value(params.vibrance)
        self.saturation_slider.set_value(params.saturation)
        self.hue_slider.set_value(params.hue_shift)
        self.clarity_slider.set_value(params.clarity)
        self.dehaze_slider.set_value(params.dehaze)
        self.vignette_slider.set_value(params.vignette)
        self.grain_slider.set_value(params.grain)
        self.fade_slider.set_value(params.fade)

        self.blockSignals(False)

    def reset_params(self):
        """重置所有参数"""
        self.set_params(ColorGradingParams())
        self.params_changed.emit(ColorGradingParams())

    def _copy_params(self):
        """复制参数到剪贴板"""
        params = self.get_params()
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(str(params.to_dict()))
    
    def set_reference_status(self, message: str, success: bool = True):
        """设置参考图片状态信息"""
        color = "#4CAF50" if success else "#888"
        self.reference_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.reference_status_label.setText(message)
