"""
调色参数数据类
独立模块，轻量导入，不依赖任何 AI 库
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ColorGradingParams:
    """调色参数数据类"""
    # 基础调整
    exposure: float = 0.0          # 曝光 [-2, 2]
    brightness: float = 0.0        # 亮度 [-1, 1]
    contrast: float = 1.0          # 对比度 [0.5, 2]
    gamma: float = 1.0             # Gamma / 中间调亮度 [0.25, 3]
    highlights: float = 0.0        # 高光 [-100, 100]
    shadows: float = 0.0           # 阴影 [-100, 100]
    whites: float = 0.0            # 白色 [-100, 100]
    blacks: float = 0.0            # 黑色 [-100, 100]

    # 颜色调整
    temperature: float = 0.0       # 色温 [-100, 100] 负值偏蓝,正值偏黄
    tint: float = 0.0              # 色调 [-100, 100] 负值偏绿,正值偏品红
    vibrance: float = 0.0          # 自然饱和度 [-100, 100]
    saturation: float = 1.0        # 饱和度 [0, 2]

    # HSL调整
    hue_shift: float = 0.0         # 色相偏移 [-180, 180]
    red_hue: float = 0.0           # 红色色相偏移 [-60, 60]
    red_saturation: float = 0.0    # 红色饱和度 [-100, 100]
    red_luminance: float = 0.0     # 红色明度 [-100, 100]
    orange_hue: float = 0.0        # 橙色色相偏移 [-60, 60]
    orange_saturation: float = 0.0
    orange_luminance: float = 0.0
    yellow_hue: float = 0.0
    yellow_saturation: float = 0.0
    yellow_luminance: float = 0.0
    green_hue: float = 0.0
    green_saturation: float = 0.0
    green_luminance: float = 0.0
    aqua_hue: float = 0.0
    aqua_saturation: float = 0.0
    aqua_luminance: float = 0.0
    blue_hue: float = 0.0
    blue_saturation: float = 0.0
    blue_luminance: float = 0.0
    purple_hue: float = 0.0
    purple_saturation: float = 0.0
    purple_luminance: float = 0.0
    magenta_hue: float = 0.0
    magenta_saturation: float = 0.0
    magenta_luminance: float = 0.0

    # 分离色调
    split_tone_shadows: List[int] = field(default_factory=lambda: [0, 0, 0])
    split_tone_highlights: List[int] = field(default_factory=lambda: [255, 255, 255])
    split_tone_balance: float = 0.0  # [-100, 100]
    shadow_hue: float = 0.0          # 阴影色轮色相 [0, 360]
    shadow_saturation: float = 0.0   # 阴影色轮强度 [0, 100]
    midtone_hue: float = 0.0         # 中间调色轮色相 [0, 360]
    midtone_saturation: float = 0.0  # 中间调色轮强度 [0, 100]
    highlight_hue: float = 0.0       # 高光色轮色相 [0, 360]
    highlight_saturation: float = 0.0  # 高光色轮强度 [0, 100]

    # 曲线与通道
    curve_shadows: float = 0.0       # 参数曲线阴影 [-100, 100]
    curve_darks: float = 0.0         # 参数曲线暗调 [-100, 100]
    curve_lights: float = 0.0        # 参数曲线亮调 [-100, 100]
    curve_highlights: float = 0.0    # 参数曲线高光 [-100, 100]
    red_balance: float = 0.0         # 红通道平衡 [-100, 100]
    green_balance: float = 0.0       # 绿通道平衡 [-100, 100]
    blue_balance: float = 0.0        # 蓝通道平衡 [-100, 100]
    cdl_slope: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    cdl_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cdl_power: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    cdl_saturation: float = 1.0

    # 效果
    clarity: float = 0.0           # 清晰度 [-100, 100]
    texture: float = 0.0           # 纹理 [-100, 100]
    midtone_detail: float = 0.0    # 中间调细节 [-100, 100]
    sharpen: float = 0.0           # 锐化 [0, 100]
    noise_reduction: float = 0.0   # 降噪 [0, 100]
    dehaze: float = 0.0            # 去雾 [-100, 100]
    bloom: float = 0.0             # 高光柔光 [0, 100]
    vignette: float = 0.0          # 暗角 [0, 100]
    grain: float = 0.0             # 颗粒 [0, 100]
    fade: float = 0.0              # 褪色 [0, 1]

    # 曲线 (可选的自定义曲线点)
    tone_curve: Optional[List[Tuple[int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorGradingParams':
        """从字典创建"""
        normalized = cls.normalize_dict(data)
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in normalized.items() if k in valid_fields})

    @classmethod
    def normalize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """归一化 LLM/旧配方可能返回的别名和嵌套参数。"""
        normalized = dict(data or {})

        # 允许 LLM 返回更接近调色软件命名的嵌套结构。
        hsl = normalized.pop('hsl', None)
        if isinstance(hsl, dict):
            for color_name, values in hsl.items():
                if not isinstance(values, dict):
                    continue
                prefix = str(color_name).lower()
                for source_key, target_suffix in (
                        ('hue', 'hue'),
                        ('hue_shift', 'hue'),
                        ('saturation', 'saturation'),
                        ('sat', 'saturation'),
                        ('luminance', 'luminance'),
                        ('lum', 'luminance')):
                    if source_key in values:
                        normalized[f'{prefix}_{target_suffix}'] = values[source_key]

        color_wheels = normalized.pop('color_wheels', None) or normalized.pop('tone_wheels', None)
        if isinstance(color_wheels, dict):
            wheel_aliases = {
                'shadow': 'shadow',
                'shadows': 'shadow',
                'midtone': 'midtone',
                'midtones': 'midtone',
                'highlight': 'highlight',
                'highlights': 'highlight',
            }
            for wheel_name, values in color_wheels.items():
                if not isinstance(values, dict):
                    continue
                prefix = wheel_aliases.get(str(wheel_name).lower())
                if not prefix:
                    continue
                if 'hue' in values:
                    normalized[f'{prefix}_hue'] = values['hue']
                if 'saturation' in values:
                    normalized[f'{prefix}_saturation'] = values['saturation']
                if 'strength' in values:
                    normalized[f'{prefix}_saturation'] = values['strength']

        cdl = normalized.pop('cdl', None)
        if isinstance(cdl, dict):
            for source_key, target_key in (
                    ('slope', 'cdl_slope'),
                    ('offset', 'cdl_offset'),
                    ('power', 'cdl_power'),
                    ('saturation', 'cdl_saturation')):
                if source_key in cdl:
                    normalized[target_key] = cdl[source_key]

        rgb_balance = normalized.pop('rgb_balance', None) or normalized.pop('channel_balance', None)
        if isinstance(rgb_balance, dict):
            for color_name in ('red', 'green', 'blue'):
                if color_name in rgb_balance:
                    normalized[f'{color_name}_balance'] = rgb_balance[color_name]

        aliases = {
            'midtones_hue': 'midtone_hue',
            'midtones_saturation': 'midtone_saturation',
            'shadow_strength': 'shadow_saturation',
            'midtone_strength': 'midtone_saturation',
            'highlight_strength': 'highlight_saturation',
            'color_boost': 'vibrance',
            'midtones': 'gamma',
        }
        for source_key, target_key in aliases.items():
            if source_key in normalized and target_key not in normalized:
                normalized[target_key] = normalized[source_key]

        valid_fields = cls.__dataclass_fields__
        return {k: v for k, v in normalized.items() if k in valid_fields}
