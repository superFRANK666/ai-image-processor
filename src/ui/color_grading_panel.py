"""
调色面板
支持自然语言输入和参数手动调整
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSlider, QGroupBox, QScrollArea, QComboBox
)
from PySide6.QtCore import Qt, Signal

# 直接从轻量模块导入，不触发 numpy / sentence_transformers
from ..ai.color_params import ColorGradingParams
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
        self.label.setMinimumWidth(96)  # 增大最小宽度以避免文字被截断
        self.label.setFixedWidth(96)  # 固定宽度确保对齐
        layout.addWidget(self.label)

        # 滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self.scale), int(max_val * self.scale))
        self.slider.setValue(int(default * self.scale))
        self.slider.valueChanged.connect(self._on_slider_changed)

        # 禁用滚轮（转发给父级滚动区域）
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
        value = max(self.min_val, min(self.max_val, float(value)))
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * self.scale))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{value:.{self.decimals}f}{self.suffix}")

    def reset(self):
        """重置到默认值"""
        self.slider.setValue(int((self.min_val + self.max_val) / 2 * self.scale))


class ColorGradingPanel(QWidget):
    """调色面板"""

    HSL_CHANNELS = (
        ("red", "红色"),
        ("orange", "橙色"),
        ("yellow", "黄色"),
        ("green", "绿色"),
        ("aqua", "青色"),
        ("blue", "蓝色"),
        ("purple", "紫色"),
        ("magenta", "品红"),
    )

    params_changed = Signal(object)  # ColorGradingParams
    text_input_submitted = Signal(str)
    find_similar_requested = Signal()
    upload_reference_requested = Signal()  # 请求上传参考图片
    look_apply_requested = Signal(str)
    save_look_requested = Signal()
    reset_all_requested = Signal()  # 删除所有调色，恢复原图

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
        self._image_available = False
        self._stored_params = ColorGradingParams()

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
        self.apply_btn.setProperty("variant", "primary")
        self.apply_btn.setMinimumWidth(60)
        input_layout.addWidget(self.apply_btn)
        nlp_layout.addLayout(input_layout)

        # 风格参考功能区
        style_ref_layout = QHBoxLayout()

        # 查找相似按钮
        self.find_similar_btn = QPushButton("查找相似风格")
        self.find_similar_btn.setProperty("variant", "secondary")
        self.find_similar_btn.setToolTip("在图像库中查找与当前图片风格相似的图片")
        style_ref_layout.addWidget(self.find_similar_btn)

        # 上传参考图片按钮
        self.upload_reference_btn = QPushButton("上传参考图片")
        self.upload_reference_btn.setProperty("variant", "secondary")
        self.upload_reference_btn.setToolTip("上传一张参考图片，提取其色调并应用到当前图片")
        style_ref_layout.addWidget(self.upload_reference_btn)

        nlp_layout.addLayout(style_ref_layout)

        # 参考图片状态标签
        self.reference_status_label = QLabel("")
        self.reference_status_label.setObjectName("inlineStatus")
        self.reference_status_label.setProperty("state", "muted")
        self.reference_status_label.setWordWrap(True)
        nlp_layout.addWidget(self.reference_status_label)

        layout.addWidget(nlp_group)

        # 可复用风格配方
        look_group = QGroupBox("风格配方")
        look_layout = QVBoxLayout(look_group)

        self.look_combo = QComboBox()
        self.look_combo.addItem("选择风格配方...", None)
        self.look_combo.setToolTip("保存和复用常用调色参数")
        self.look_combo.installEventFilter(self._wheel_blocker)
        look_layout.addWidget(self.look_combo)

        look_btn_layout = QHBoxLayout()
        self.look_apply_btn = QPushButton("套用")
        self.look_apply_btn.setProperty("variant", "primary")
        self.look_apply_btn.setToolTip("将所选风格配方应用到当前图像")
        look_btn_layout.addWidget(self.look_apply_btn)

        self.save_look_btn = QPushButton("保存当前")
        self.save_look_btn.setProperty("variant", "secondary")
        self.save_look_btn.setToolTip("把当前调色参数保存为可复用风格配方")
        look_btn_layout.addWidget(self.save_look_btn)

        look_layout.addLayout(look_btn_layout)

        # 恢复原图按钮（删除所有调色，恢复照片原始状态）
        self.reset_to_original_btn = QPushButton("恢复原图")
        self.reset_to_original_btn.setProperty("variant", "danger")
        self.reset_to_original_btn.setToolTip("删除所有调色，将照片恢复到原始状态")
        look_layout.addWidget(self.reset_to_original_btn)

        self.look_status_label = QLabel("内置配方可直接套用，自定义配方会保存到本地。")
        self.look_status_label.setObjectName("inlineStatus")
        self.look_status_label.setProperty("state", "muted")
        self.look_status_label.setWordWrap(True)
        look_layout.addWidget(self.look_status_label)

        layout.addWidget(look_group)

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

        self.brightness_slider = ParamSlider("亮度", -1.0, 1.0, 0.0, 2)
        basic_layout.addWidget(self.brightness_slider)

        self.contrast_slider = ParamSlider("对比度", 0.5, 2.0, 1.0, 2)
        basic_layout.addWidget(self.contrast_slider)

        self.gamma_slider = ParamSlider("中间调", 0.25, 3.0, 1.0, 2)
        basic_layout.addWidget(self.gamma_slider)

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

        self.red_balance_slider = ParamSlider("红通道", -100, 100, 0, 0)
        color_layout.addWidget(self.red_balance_slider)

        self.green_balance_slider = ParamSlider("绿通道", -100, 100, 0, 0)
        color_layout.addWidget(self.green_balance_slider)

        self.blue_balance_slider = ParamSlider("蓝通道", -100, 100, 0, 0)
        color_layout.addWidget(self.blue_balance_slider)

        params_layout.addWidget(color_group)

        # HSL 分色
        hsl_group = QGroupBox("HSL 分色")
        hsl_layout = QVBoxLayout(hsl_group)
        self._hsl_param_sliders = []

        for color_key, color_label in self.HSL_CHANNELS:
            hue_slider = ParamSlider(f"{color_label}色相", -60, 60, 0, 0, "°")
            saturation_slider = ParamSlider(f"{color_label}饱和", -100, 100, 0, 0)
            luminance_slider = ParamSlider(f"{color_label}明度", -100, 100, 0, 0)

            setattr(self, f"{color_key}_hue_slider", hue_slider)
            setattr(self, f"{color_key}_saturation_slider", saturation_slider)
            setattr(self, f"{color_key}_luminance_slider", luminance_slider)

            hsl_layout.addWidget(hue_slider)
            hsl_layout.addWidget(saturation_slider)
            hsl_layout.addWidget(luminance_slider)
            self._hsl_param_sliders.extend([hue_slider, saturation_slider, luminance_slider])

        params_layout.addWidget(hsl_group)

        # 曲线
        curve_group = QGroupBox("参数曲线")
        curve_layout = QVBoxLayout(curve_group)

        self.curve_shadows_slider = ParamSlider("曲线阴影", -100, 100, 0, 0)
        curve_layout.addWidget(self.curve_shadows_slider)

        self.curve_darks_slider = ParamSlider("暗调", -100, 100, 0, 0)
        curve_layout.addWidget(self.curve_darks_slider)

        self.curve_lights_slider = ParamSlider("亮调", -100, 100, 0, 0)
        curve_layout.addWidget(self.curve_lights_slider)

        self.curve_highlights_slider = ParamSlider("曲线高光", -100, 100, 0, 0)
        curve_layout.addWidget(self.curve_highlights_slider)

        params_layout.addWidget(curve_group)

        # 色轮
        wheels_group = QGroupBox("三路色轮")
        wheels_layout = QVBoxLayout(wheels_group)

        self.shadow_hue_slider = ParamSlider("阴影色相", 0, 360, 0, 0, "°")
        wheels_layout.addWidget(self.shadow_hue_slider)

        self.shadow_saturation_slider = ParamSlider("阴影强度", 0, 100, 0, 0)
        wheels_layout.addWidget(self.shadow_saturation_slider)

        self.midtone_hue_slider = ParamSlider("中调色相", 0, 360, 0, 0, "°")
        wheels_layout.addWidget(self.midtone_hue_slider)

        self.midtone_saturation_slider = ParamSlider("中调强度", 0, 100, 0, 0)
        wheels_layout.addWidget(self.midtone_saturation_slider)

        self.highlight_hue_slider = ParamSlider("高光色相", 0, 360, 0, 0, "°")
        wheels_layout.addWidget(self.highlight_hue_slider)

        self.highlight_saturation_slider = ParamSlider("高光强度", 0, 100, 0, 0)
        wheels_layout.addWidget(self.highlight_saturation_slider)

        params_layout.addWidget(wheels_group)

        # 效果
        effects_group = QGroupBox("效果")
        effects_layout = QVBoxLayout(effects_group)

        self.clarity_slider = ParamSlider("清晰度", -100, 100, 0, 0)
        effects_layout.addWidget(self.clarity_slider)

        self.texture_slider = ParamSlider("纹理", -100, 100, 0, 0)
        effects_layout.addWidget(self.texture_slider)

        self.midtone_detail_slider = ParamSlider("中调细节", -100, 100, 0, 0)
        effects_layout.addWidget(self.midtone_detail_slider)

        self.sharpen_slider = ParamSlider("锐化", 0, 100, 0, 0)
        effects_layout.addWidget(self.sharpen_slider)

        self.noise_reduction_slider = ParamSlider("降噪", 0, 100, 0, 0)
        effects_layout.addWidget(self.noise_reduction_slider)

        self.dehaze_slider = ParamSlider("去雾", -100, 100, 0, 0)
        effects_layout.addWidget(self.dehaze_slider)

        self.bloom_slider = ParamSlider("柔光", 0, 100, 0, 0)
        effects_layout.addWidget(self.bloom_slider)

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
        self.reset_btn.setProperty("variant", "secondary")
        btn_layout.addWidget(self.reset_btn)

        self.copy_params_btn = QPushButton("复制参数")
        self.copy_params_btn.setProperty("variant", "secondary")
        btn_layout.addWidget(self.copy_params_btn)

        layout.addLayout(btn_layout)

        self._image_dependent_widgets = [
            self.text_input,
            self.apply_btn,
            self.look_combo,
            self.find_similar_btn,
            self.upload_reference_btn,
            self.reset_btn,
            self.copy_params_btn,
            self.exposure_slider,
            self.brightness_slider,
            self.contrast_slider,
            self.gamma_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.whites_slider,
            self.blacks_slider,
            self.temperature_slider,
            self.tint_slider,
            self.vibrance_slider,
            self.saturation_slider,
            self.hue_slider,
            self.red_balance_slider,
            self.green_balance_slider,
            self.blue_balance_slider,
            *self._hsl_param_sliders,
            self.curve_shadows_slider,
            self.curve_darks_slider,
            self.curve_lights_slider,
            self.curve_highlights_slider,
            self.shadow_hue_slider,
            self.shadow_saturation_slider,
            self.midtone_hue_slider,
            self.midtone_saturation_slider,
            self.highlight_hue_slider,
            self.highlight_saturation_slider,
            self.clarity_slider,
            self.texture_slider,
            self.midtone_detail_slider,
            self.sharpen_slider,
            self.noise_reduction_slider,
            self.dehaze_slider,
            self.bloom_slider,
            self.vignette_slider,
            self.grain_slider,
            self.fade_slider,
        ]

    def _connect_signals(self):
        """连接信号"""
        # 文本输入
        self.text_input.returnPressed.connect(self._on_text_submitted)
        self.apply_btn.clicked.connect(self._on_text_submitted)

        # 查找相似
        self.find_similar_btn.clicked.connect(self.find_similar_requested.emit)

        # 上传参考图片
        self.upload_reference_btn.clicked.connect(self.upload_reference_requested.emit)

        # 滑块变化
        self.exposure_slider.value_changed.connect(self._on_param_changed)
        self.brightness_slider.value_changed.connect(self._on_param_changed)
        self.contrast_slider.value_changed.connect(self._on_param_changed)
        self.gamma_slider.value_changed.connect(self._on_param_changed)
        self.highlights_slider.value_changed.connect(self._on_param_changed)
        self.shadows_slider.value_changed.connect(self._on_param_changed)
        self.whites_slider.value_changed.connect(self._on_param_changed)
        self.blacks_slider.value_changed.connect(self._on_param_changed)
        self.temperature_slider.value_changed.connect(self._on_param_changed)
        self.tint_slider.value_changed.connect(self._on_param_changed)
        self.vibrance_slider.value_changed.connect(self._on_param_changed)
        self.saturation_slider.value_changed.connect(self._on_param_changed)
        self.hue_slider.value_changed.connect(self._on_param_changed)
        self.red_balance_slider.value_changed.connect(self._on_param_changed)
        self.green_balance_slider.value_changed.connect(self._on_param_changed)
        self.blue_balance_slider.value_changed.connect(self._on_param_changed)
        for slider in self._hsl_param_sliders:
            slider.value_changed.connect(self._on_param_changed)
        self.curve_shadows_slider.value_changed.connect(self._on_param_changed)
        self.curve_darks_slider.value_changed.connect(self._on_param_changed)
        self.curve_lights_slider.value_changed.connect(self._on_param_changed)
        self.curve_highlights_slider.value_changed.connect(self._on_param_changed)
        self.shadow_hue_slider.value_changed.connect(self._on_param_changed)
        self.shadow_saturation_slider.value_changed.connect(self._on_param_changed)
        self.midtone_hue_slider.value_changed.connect(self._on_param_changed)
        self.midtone_saturation_slider.value_changed.connect(self._on_param_changed)
        self.highlight_hue_slider.value_changed.connect(self._on_param_changed)
        self.highlight_saturation_slider.value_changed.connect(self._on_param_changed)
        self.clarity_slider.value_changed.connect(self._on_param_changed)
        self.texture_slider.value_changed.connect(self._on_param_changed)
        self.midtone_detail_slider.value_changed.connect(self._on_param_changed)
        self.sharpen_slider.value_changed.connect(self._on_param_changed)
        self.noise_reduction_slider.value_changed.connect(self._on_param_changed)
        self.dehaze_slider.value_changed.connect(self._on_param_changed)
        self.bloom_slider.value_changed.connect(self._on_param_changed)
        self.vignette_slider.value_changed.connect(self._on_param_changed)
        self.grain_slider.value_changed.connect(self._on_param_changed)
        self.fade_slider.value_changed.connect(self._on_param_changed)

        # 重置按钮
        self.reset_btn.clicked.connect(lambda: self.reset_params())

        # 复制参数
        self.copy_params_btn.clicked.connect(self._copy_params)

        # 风格配方
        self.look_combo.currentIndexChanged.connect(self._update_look_actions)
        self.look_apply_btn.clicked.connect(self._on_apply_look)
        self.save_look_btn.clicked.connect(self.save_look_requested.emit)
        # 恢复原图：清除所有调色并恢复到照片原始状态
        self.reset_to_original_btn.clicked.connect(self.reset_all_requested.emit)
        self._update_look_actions()

    def _on_text_submitted(self):
        """文本提交"""
        text = self.text_input.text().strip()
        if text:
            self.text_input_submitted.emit(text)

    def _on_param_changed(self, value: float):
        """参数改变 - 使用防抖机制"""
        # 重启防抖定时器，只有在用户停止调节后才触发处理
        self._debounce_timer.start()

    def _emit_params_changed(self):
        """实际发送参数变化信号"""
        params = self.get_params()
        self._stored_params = params
        self.params_changed.emit(params)

    def get_params(self) -> ColorGradingParams:
        """获取当前参数"""
        params = self._stored_params.to_dict()
        params.update({
            "exposure": self.exposure_slider.get_value(),
            "brightness": self.brightness_slider.get_value(),
            "contrast": self.contrast_slider.get_value(),
            "gamma": self.gamma_slider.get_value(),
            "highlights": self.highlights_slider.get_value(),
            "shadows": self.shadows_slider.get_value(),
            "whites": self.whites_slider.get_value(),
            "blacks": self.blacks_slider.get_value(),
            "temperature": self.temperature_slider.get_value(),
            "tint": self.tint_slider.get_value(),
            "vibrance": self.vibrance_slider.get_value(),
            "saturation": self.saturation_slider.get_value(),
            "hue_shift": self.hue_slider.get_value(),
            "red_balance": self.red_balance_slider.get_value(),
            "green_balance": self.green_balance_slider.get_value(),
            "blue_balance": self.blue_balance_slider.get_value(),
            "curve_shadows": self.curve_shadows_slider.get_value(),
            "curve_darks": self.curve_darks_slider.get_value(),
            "curve_lights": self.curve_lights_slider.get_value(),
            "curve_highlights": self.curve_highlights_slider.get_value(),
            "shadow_hue": self.shadow_hue_slider.get_value(),
            "shadow_saturation": self.shadow_saturation_slider.get_value(),
            "midtone_hue": self.midtone_hue_slider.get_value(),
            "midtone_saturation": self.midtone_saturation_slider.get_value(),
            "highlight_hue": self.highlight_hue_slider.get_value(),
            "highlight_saturation": self.highlight_saturation_slider.get_value(),
            "clarity": self.clarity_slider.get_value(),
            "texture": self.texture_slider.get_value(),
            "midtone_detail": self.midtone_detail_slider.get_value(),
            "sharpen": self.sharpen_slider.get_value(),
            "noise_reduction": self.noise_reduction_slider.get_value(),
            "dehaze": self.dehaze_slider.get_value(),
            "bloom": self.bloom_slider.get_value(),
            "vignette": self.vignette_slider.get_value(),
            "grain": self.grain_slider.get_value(),
            "fade": self.fade_slider.get_value(),
        })
        for color_key, _ in self.HSL_CHANNELS:
            params.update({
                f"{color_key}_hue": getattr(self, f"{color_key}_hue_slider").get_value(),
                f"{color_key}_saturation": getattr(self, f"{color_key}_saturation_slider").get_value(),
                f"{color_key}_luminance": getattr(self, f"{color_key}_luminance_slider").get_value(),
            })
        return ColorGradingParams.from_dict(params)

    def set_params(self, params: ColorGradingParams):
        """设置参数"""
        if isinstance(params, dict):
            params = ColorGradingParams.from_dict(params)
        self._stored_params = params

        # 阻止信号发送以避免循环
        self.blockSignals(True)

        self.exposure_slider.set_value(params.exposure)
        self.brightness_slider.set_value(params.brightness)
        self.contrast_slider.set_value(params.contrast)
        self.gamma_slider.set_value(params.gamma)
        self.highlights_slider.set_value(params.highlights)
        self.shadows_slider.set_value(params.shadows)
        self.whites_slider.set_value(params.whites)
        self.blacks_slider.set_value(params.blacks)
        self.temperature_slider.set_value(params.temperature)
        self.tint_slider.set_value(params.tint)
        self.vibrance_slider.set_value(params.vibrance)
        self.saturation_slider.set_value(params.saturation)
        self.hue_slider.set_value(params.hue_shift)
        self.red_balance_slider.set_value(params.red_balance)
        self.green_balance_slider.set_value(params.green_balance)
        self.blue_balance_slider.set_value(params.blue_balance)
        for color_key, _ in self.HSL_CHANNELS:
            getattr(self, f"{color_key}_hue_slider").set_value(getattr(params, f"{color_key}_hue"))
            getattr(self, f"{color_key}_saturation_slider").set_value(getattr(params, f"{color_key}_saturation"))
            getattr(self, f"{color_key}_luminance_slider").set_value(getattr(params, f"{color_key}_luminance"))
        self.curve_shadows_slider.set_value(params.curve_shadows)
        self.curve_darks_slider.set_value(params.curve_darks)
        self.curve_lights_slider.set_value(params.curve_lights)
        self.curve_highlights_slider.set_value(params.curve_highlights)
        self.shadow_hue_slider.set_value(params.shadow_hue)
        self.shadow_saturation_slider.set_value(params.shadow_saturation)
        self.midtone_hue_slider.set_value(params.midtone_hue)
        self.midtone_saturation_slider.set_value(params.midtone_saturation)
        self.highlight_hue_slider.set_value(params.highlight_hue)
        self.highlight_saturation_slider.set_value(params.highlight_saturation)
        self.clarity_slider.set_value(params.clarity)
        self.texture_slider.set_value(params.texture)
        self.midtone_detail_slider.set_value(params.midtone_detail)
        self.sharpen_slider.set_value(params.sharpen)
        self.noise_reduction_slider.set_value(params.noise_reduction)
        self.dehaze_slider.set_value(params.dehaze)
        self.bloom_slider.set_value(params.bloom)
        self.vignette_slider.set_value(params.vignette)
        self.grain_slider.set_value(params.grain)
        self.fade_slider.set_value(params.fade)

        self.blockSignals(False)

    def reset_params(self, emit_change: bool = True):
        """重置所有参数"""
        self._debounce_timer.stop()
        params = ColorGradingParams()
        self.set_params(params)
        if emit_change:
            self.params_changed.emit(params)

    def _copy_params(self):
        """复制参数到剪贴板"""
        params = self.get_params()
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(str(params.to_dict()))

    def set_reference_status(self, message: str, success: bool = True):
        """设置参考图片状态信息"""
        self._set_status_label_state(self.reference_status_label, "success" if success else "muted")
        self.reference_status_label.setText(message)

    def set_image_available(self, available: bool):
        """根据是否有当前图像启用或禁用调色交互。"""
        self._image_available = available
        for widget in self._image_dependent_widgets:
            widget.setEnabled(available)
        self._update_look_actions()

    def set_look_presets(self, presets, selected_id: Optional[str] = None):
        """刷新可选风格配方。"""
        current_id = selected_id or self.get_selected_look_id()
        self.look_combo.blockSignals(True)
        self.look_combo.clear()
        self.look_combo.addItem("选择风格配方...", None)
        for preset in presets:
            if hasattr(preset, "to_dict"):
                preset = preset.to_dict()
            source = preset.get("source", "custom")
            suffix = "内置" if source == "builtin" else "自定义"
            self.look_combo.addItem(
                f"{preset.get('name', '未命名')} · {suffix}",
                {
                    "id": preset.get("id"),
                    "source": source,
                    "description": preset.get("description", ""),
                },
            )
        if current_id:
            for index in range(self.look_combo.count()):
                data = self.look_combo.itemData(index)
                if isinstance(data, dict) and data.get("id") == current_id:
                    self.look_combo.setCurrentIndex(index)
                    break
        self.look_combo.blockSignals(False)
        self._update_look_actions()

    def get_selected_look_id(self):
        """获取当前选中的风格配方 id。"""
        data = self.look_combo.currentData()
        if isinstance(data, dict):
            return data.get("id")
        return None

    def set_look_status(self, message: str, success: bool = True):
        """设置风格配方状态信息。"""
        self._set_status_label_state(self.look_status_label, "success" if success else "error")
        self.look_status_label.setText(message)

    def _on_apply_look(self):
        preset_id = self.get_selected_look_id()
        if preset_id:
            self.look_apply_requested.emit(preset_id)

    def _update_look_actions(self):
        data = self.look_combo.currentData() if hasattr(self, "look_combo") else None
        has_preset = isinstance(data, dict) and bool(data.get("id"))
        self.look_apply_btn.setEnabled(self._image_available and has_preset)
        self.save_look_btn.setEnabled(self._image_available)
        # 恢复原图按钮仅在有图像时可用
        self.reset_to_original_btn.setEnabled(self._image_available)

    def _set_status_label_state(self, label: QLabel, state: str):
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)
