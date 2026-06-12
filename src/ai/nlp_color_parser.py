"""
自然语言调色参数解析器
将用户的自然语言描述转换为具体的调色参数
"""
import re
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np

# ColorGradingParams 定义在独立的轻量模块中，UI 面板导入本模块时不触发 numpy 加载
from .color_params import ColorGradingParams  # noqa: F401 (re-export)


class NLPColorParser:
    """
    自然语言调色解析器
    将用户的文字描述转换为调色参数
    """

    # 颜色关键词映射
    COLOR_KEYWORDS = {
        # 色调关键词
        '蓝调': {'temperature': -25, 'tint': -5, 'saturation': 1.1},
        '青蓝': {'temperature': -30, 'hue_shift': -15},
        '冷色': {'temperature': -20},
        '暖色': {'temperature': 25},
        '暖调': {'temperature': 20, 'saturation': 1.05},
        '金色': {'temperature': 35, 'saturation': 1.15},
        '金黄': {
            'temperature': 28, 'vibrance': 14,
            'orange_saturation': 16, 'yellow_saturation': 20,
            'yellow_luminance': 8, 'highlight_hue': 42,
            'highlight_saturation': 24,
        },
        '黄色': {
            'temperature': 18, 'yellow_saturation': 22,
            'yellow_luminance': 6, 'highlight_hue': 48,
            'highlight_saturation': 14,
        },
        '橙调': {'temperature': 30, 'hue_shift': 15},
        '绿调': {'tint': -20, 'hue_shift': 60},
        '深绿色': {
            'exposure': -0.08, 'gamma': 0.92, 'tint': -18,
            'green_hue': -8, 'green_saturation': 34,
            'green_luminance': -14, 'yellow_saturation': -16,
            'curve_darks': -8, 'midtone_hue': 132,
            'midtone_saturation': 18,
        },
        '绿色': {
            'tint': -14, 'green_hue': -6, 'green_saturation': 26,
            'green_luminance': -4, 'yellow_saturation': -10,
            'midtone_hue': 128, 'midtone_saturation': 10,
        },
        '蓝色': {
            'temperature': -18, 'blue_saturation': 26,
            'blue_luminance': -6, 'aqua_saturation': 10,
            'shadow_hue': 220, 'shadow_saturation': 12,
        },
        '红色': {
            'temperature': 10, 'red_saturation': 24,
            'red_luminance': 4, 'orange_saturation': 8,
            'highlight_hue': 20, 'highlight_saturation': 12,
        },
        '青色': {
            'temperature': -12, 'aqua_saturation': 24,
            'aqua_luminance': 5, 'blue_saturation': 10,
            'shadow_hue': 188, 'shadow_saturation': 12,
        },
        '粉调': {'tint': 25, 'saturation': 0.95},
        '粉色': {
            'tint': 22, 'red_saturation': 10, 'orange_luminance': 8,
            'magenta_saturation': 18, 'highlight_hue': 330,
            'highlight_saturation': 12,
        },
        '紫调': {'hue_shift': -30, 'tint': 15},
        '紫色': {
            'tint': 15, 'purple_saturation': 26,
            'magenta_saturation': 12, 'shadow_hue': 270,
            'shadow_saturation': 14,
        },

        # 风格关键词
        '复古': {'saturation': 0.85, 'contrast': 1.1, 'fade': 0.12, 'grain': 15},
        '怀旧': {'saturation': 0.8, 'temperature': 10, 'fade': 0.15, 'contrast': 0.95},
        '胶片': {'contrast': 1.15, 'saturation': 0.9, 'grain': 20, 'fade': 0.1},
        '电影': {'contrast': 1.2, 'saturation': 0.88, 'shadows': -10, 'highlights': -8},
        '电影感': {'contrast': 1.25, 'saturation': 0.85, 'blacks': 5},
        '日系': {'exposure': 0.08, 'contrast': 0.92, 'saturation': 0.82, 'highlights': 10},
        '清新': {'exposure': 0.1, 'saturation': 0.9, 'vibrance': 15, 'clarity': 10},
        '森系': {'temperature': -5, 'tint': -10, 'saturation': 0.95, 'vibrance': 20},
        '黑金': {'saturation': 0.6, 'contrast': 1.3, 'temperature': 15},
        '赛博朋克': {'contrast': 1.4, 'saturation': 1.3, 'vibrance': 30},
        'ins风': {'contrast': 1.1, 'saturation': 0.85, 'fade': 0.08},
        '港风': {'contrast': 1.2, 'saturation': 1.1, 'grain': 10},
        '青橙': {
            'contrast': 1.22, 'saturation': 0.9, 'curve_shadows': -10,
            'shadow_hue': 195, 'shadow_saturation': 26,
            'highlight_hue': 38, 'highlight_saturation': 22,
        },
        'teal orange': {
            'contrast': 1.22, 'saturation': 0.9,
            'shadow_hue': 195, 'shadow_saturation': 26,
            'highlight_hue': 38, 'highlight_saturation': 22,
        },
        '大片': {'contrast': 1.25, 'curve_darks': -10, 'curve_lights': 10, 'vignette': 10},
        '低调': {'exposure': -0.2, 'gamma': 0.82, 'contrast': 1.2, 'blacks': -15, 'vignette': 18},
        '高调': {'exposure': 0.28, 'gamma': 1.15, 'contrast': 0.9, 'shadows': 15},
        '暗黑': {'exposure': -0.35, 'gamma': 0.78, 'contrast': 1.28, 'blacks': -22, 'vignette': 25},
        '莫兰迪': {'saturation': 0.68, 'contrast': 0.95, 'fade': 0.12, 'curve_shadows': 8},
        '马卡龙': {'exposure': 0.18, 'contrast': 0.88, 'saturation': 0.78, 'vibrance': 12, 'curve_shadows': 10},
        '奶油': {'exposure': 0.16, 'contrast': 0.86, 'temperature': 8, 'texture': -18, 'bloom': 8},
        '奶油肌': {'orange_luminance': 12, 'red_saturation': -10, 'texture': -25, 'noise_reduction': 18},
        '宝丽来': {'contrast': 1.08, 'saturation': 0.82, 'fade': 0.22, 'grain': 24, 'curve_shadows': 14},
        'kodak': {'temperature': 12, 'contrast': 1.12, 'orange_saturation': 12, 'blue_saturation': -8, 'grain': 18},
        '富士': {'temperature': -4, 'tint': -6, 'green_saturation': 12, 'red_saturation': -8, 'grain': 12},
        'fujifilm': {'temperature': -4, 'tint': -6, 'green_saturation': 12, 'red_saturation': -8, 'grain': 12},
        '宝莱坞': {'contrast': 1.18, 'saturation': 1.22, 'vibrance': 26, 'temperature': 12},
        '黑白': {'saturation': 0.0, 'contrast': 1.2, 'curve_darks': -8, 'curve_lights': 8},
        '单色': {'saturation': 0.18, 'contrast': 1.12},
        '银盐': {'saturation': 0.0, 'contrast': 1.28, 'grain': 28, 'curve_shadows': 10},
        '霓虹': {
            'contrast': 1.28, 'saturation': 1.24, 'vibrance': 35, 'bloom': 22,
            'shadow_hue': 245, 'shadow_saturation': 22,
            'highlight_hue': 305, 'highlight_saturation': 26,
        },
        '蒸汽波': {
            'contrast': 1.08, 'saturation': 1.28, 'bloom': 18,
            'shadow_hue': 220, 'shadow_saturation': 18,
            'highlight_hue': 315, 'highlight_saturation': 28,
        },
        'vaporwave': {
            'contrast': 1.08, 'saturation': 1.28, 'bloom': 18,
            'shadow_hue': 220, 'shadow_saturation': 18,
            'highlight_hue': 315, 'highlight_saturation': 28,
        },

        # 场景关键词
        '海边': {'temperature': -15, 'saturation': 1.1, 'vibrance': 15, 'dehaze': 10},
        '海滩': {'temperature': -10, 'saturation': 1.15, 'clarity': 15},
        '夕阳': {'temperature': 40, 'saturation': 1.2, 'vibrance': 20},
        '日落': {'temperature': 35, 'saturation': 1.15, 'highlights': -10},
        '夜景': {'contrast': 1.3, 'blacks': -10, 'highlights': -15},
        '城市夜景': {'contrast': 1.35, 'saturation': 1.1, 'clarity': 20},
        '森林': {'tint': -15, 'saturation': 1.1, 'vibrance': 25},
        '雪景': {'temperature': -10, 'exposure': 0.15, 'contrast': 1.1},
        '秋天': {'temperature': 20, 'saturation': 1.2, 'vibrance': 15},
        '春天': {'saturation': 1.1, 'vibrance': 20, 'clarity': 10},
        '蓝天': {'blue_saturation': 28, 'blue_luminance': -6, 'aqua_saturation': 14, 'dehaze': 12},
        '天空': {'blue_saturation': 22, 'blue_luminance': -4, 'aqua_saturation': 10, 'dehaze': 10},
        '云': {'highlights': 8, 'whites': 10, 'dehaze': 8},
        '草地': {'green_saturation': 24, 'green_luminance': -4, 'yellow_saturation': -8},
        '树叶': {'green_saturation': 18, 'green_hue': -6, 'yellow_saturation': -6},
        '肤色': {'orange_luminance': 10, 'orange_saturation': 6, 'red_saturation': -8, 'texture': -10},
        '人像': {'orange_luminance': 8, 'red_saturation': -8, 'texture': -12, 'clarity': -6},
        '黄金时刻': {
            'temperature': 22, 'highlight_hue': 40, 'highlight_saturation': 25,
            'orange_saturation': 14, 'yellow_luminance': 8,
        },
        '金色时刻': {
            'temperature': 22, 'highlight_hue': 40, 'highlight_saturation': 25,
            'orange_saturation': 14, 'yellow_luminance': 8,
        },
        '蓝调时刻': {
            'temperature': -22, 'gamma': 0.9, 'blue_saturation': 18,
            'shadow_hue': 218, 'shadow_saturation': 18,
        },
        '雨天': {'temperature': -12, 'contrast': 0.95, 'dehaze': -8, 'blue_saturation': 10},
        '雾': {'dehaze': -25, 'contrast': 0.86, 'bloom': 10, 'fade': 0.08},
        '沙漠': {'temperature': 24, 'yellow_saturation': 18, 'orange_saturation': 10, 'dehaze': 10},
        '雪': {'temperature': -12, 'exposure': 0.18, 'whites': 12, 'blue_saturation': -8},

        # 情绪关键词
        '明亮': {'exposure': 0.2, 'highlights': 15},
        '阴暗': {'exposure': -0.15, 'shadows': -20},
        '柔和': {'contrast': 0.9, 'clarity': -15},
        '锐利': {'contrast': 1.15, 'clarity': 30},
        '梦幻': {'contrast': 0.85, 'saturation': 0.9, 'fade': 0.2, 'clarity': -20},
        '通透': {'dehaze': 25, 'clarity': 20, 'vibrance': 10},
        '高级感': {'contrast': 1.1, 'saturation': 0.85, 'clarity': 15},
        '质感': {'clarity': 25, 'contrast': 1.1},
        '干净': {'noise_reduction': 15, 'dehaze': 12, 'grain': 0, 'texture': -4},
        '油润': {'contrast': 1.05, 'curve_shadows': 8, 'curve_lights': 6, 'texture': -6},
        '空气感': {'exposure': 0.18, 'contrast': 0.88, 'shadows': 15, 'dehaze': -5, 'bloom': 8},
        '朦胧感': {'clarity': -18, 'texture': -14, 'dehaze': -18, 'bloom': 16},
        '柔焦': {'clarity': -22, 'texture': -20, 'bloom': 18},
        '硬朗': {'contrast': 1.22, 'clarity': 28, 'midtone_detail': 20, 'texture': 12},
        '厚重': {'contrast': 1.18, 'gamma': 0.9, 'curve_darks': -12, 'saturation': 0.9},
        '轻盈': {'exposure': 0.16, 'contrast': 0.9, 'curve_shadows': 12, 'saturation': 0.86},
    }

    # 程度修饰词
    INTENSITY_MODIFIERS = {
        '很': 1.5,
        '非常': 1.8,
        '极其': 2.0,
        '稍微': 0.5,
        '略微': 0.6,
        '一点': 0.4,
        '些许': 0.5,
        '强烈': 1.7,
        '轻微': 0.4,
        '深': 1.3,
        '浅': 0.7,
        '淡': 0.6,
        '浓': 1.4,
    }

    # 动作关键词
    ACTION_KEYWORDS = {
        '增加': 1.0,
        '提高': 1.0,
        '加强': 1.2,
        '增强': 1.2,
        '减少': -1.0,
        '降低': -1.0,
        '减弱': -0.8,
        '去除': -1.5,
    }

    # 参数关键词映射
    PARAM_KEYWORDS = {
        '曝光': 'exposure',
        '亮度': 'exposure',
        '明度': 'brightness',
        '中间调亮度': 'gamma',
        'gamma': 'gamma',
        '伽马': 'gamma',
        '对比度': 'contrast',
        '对比': 'contrast',
        '饱和度': 'saturation',
        '饱和': 'saturation',
        '色温': 'temperature',
        '色调': 'tint',
        '清晰度': 'clarity',
        '锐度': 'clarity',
        '纹理': 'texture',
        '细节': 'midtone_detail',
        '中间调细节': 'midtone_detail',
        '锐化': 'sharpen',
        '降噪': 'noise_reduction',
        '噪声': 'noise_reduction',
        '去雾': 'dehaze',
        '雾感': 'dehaze',
        '柔光': 'bloom',
        '泛光': 'bloom',
        '光晕': 'bloom',
        '发光': 'bloom',
        '暗角': 'vignette',
        '颗粒': 'grain',
        '噪点': 'grain',
        '高光': 'highlights',
        '阴影': 'shadows',
        '白色': 'whites',
        '黑色': 'blacks',
        '自然饱和度': 'vibrance',
        '鲜艳度': 'vibrance',
        '曲线阴影': 'curve_shadows',
        '暗调': 'curve_darks',
        '亮调': 'curve_lights',
        '曲线高光': 'curve_highlights',
        '红色': 'red_saturation',
        '橙色': 'orange_saturation',
        '黄色': 'yellow_saturation',
        '绿色': 'green_saturation',
        '青色': 'aqua_saturation',
        '蓝色': 'blue_saturation',
        '紫色': 'purple_saturation',
        '品红': 'magenta_saturation',
    }

    def __init__(self, use_llm: bool = True, llm_config: Optional[Dict[str, Any]] = None):
        """
        初始化解析器

        Args:
            use_llm: 是否使用大模型分析（推荐）
            llm_config: 大模型配置，例如：
                {"provider": "openai", "api_key": "sk-xxx", "model": "gpt-3.5-turbo"}
                或 {"provider": "ollama", "model": "qwen2.5:7b"}
        """
        self.text_encoder = None
        self.use_llm = use_llm
        self.llm_analyzer = None

        self._init_text_encoder()

        if use_llm:
            self._init_llm_analyzer(llm_config or {})

    def _init_text_encoder(self):
        """初始化文本编码器用于语义理解"""
        try:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path

            # 优先使用本地模型
            local_model_path = Path(__file__).parent.parent.parent / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
            if local_model_path.exists() and (local_model_path / "model.safetensors").exists():
                print(f"加载本地模型: {local_model_path}")
                self.text_encoder = SentenceTransformer(str(local_model_path))
            else:
                # 尝试从网络加载
                print("本地模型不存在，尝试从网络加载...")
                self.text_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            print(f"警告: 无法加载文本编码器: {e}")
            self.text_encoder = None

    def _init_llm_analyzer(self, llm_config: Dict[str, Any]):
        """初始化本地大模型分析器"""
        try:
            from .local_llm_analyzer import LocalLLMColorAnalyzer
            from .async_llm_analyzer import AsyncLLMColorAnalyzer

            model_name = llm_config.get('model_name', 'Qwen/Qwen2.5-1.5B-Instruct')
            device = llm_config.get('device', 'auto')

            # 创建本地分析器
            local_analyzer = LocalLLMColorAnalyzer(
                model_name=model_name,
                device=device,
                quantization_config=llm_config.get('quantization'),
                max_memory=llm_config.get('max_memory'),
                offload_folder=llm_config.get('offload_folder'),
                trust_remote_code=llm_config.get('trust_remote_code', False)
            )

            # 用异步包装器包装
            self.llm_analyzer = AsyncLLMColorAnalyzer(local_analyzer)

            if self.llm_analyzer:
                print(f"✓ 本地大模型已启用（异步模式）: {model_name}")
            else:
                print("✗ 本地模型初始化失败，将使用传统关键词匹配")
                self.use_llm = False

        except Exception as e:
            print(f"警告: 无法初始化本地大模型: {e}")
            print("  将使用传统关键词匹配")
            self.llm_analyzer = None
            self.use_llm = False

    def parse(self, text: str, reference_params: Optional[ColorGradingParams] = None) -> ColorGradingParams:
        """
        解析自然语言描述生成调色参数

        Args:
            text: 用户输入的自然语言描述
            reference_params: 参考图像的调色参数(用于"复刻"类请求)

        Returns:
            ColorGradingParams: 生成的调色参数
        """
        params = ColorGradingParams()
        original_text = text
        text_lower = text.lower().strip()

        print(f"[调色指令] 处理: {original_text}")

        # 检测是否是复刻请求
        if self._is_reference_request(text_lower) and reference_params:
            params = self._merge_params(params, reference_params)
            return params

        # 优先使用大模型分析
        if self.use_llm and self.llm_analyzer:
            try:
                result = self.llm_analyzer.analyze(original_text)

                if result.get("is_color_related", False):
                    print(f"[LLM分析] {result.get('reasoning', '无推理信息')}")

                    # 将LLM返回的参数合并到params
                    llm_params = result.get("parameters", {})
                    if llm_params:
                        params = self._merge_params(params, llm_params)
                        if not self._is_default_params(params):
                            print(f"[LLM分析] 参数: {self._format_params(params)}")
                            return params
                        print("[LLM分析] 返回了默认参数，回退到传统匹配")
                    else:
                        print("[LLM分析] 未返回参数，回退到传统匹配")
                else:
                    print(f"[LLM分析] 非调色指令: {result.get('reasoning', '')}，尝试传统匹配")

            except Exception as e:
                print(f"[LLM分析] 失败: {e}，回退到传统匹配")

        # 回退到传统的关键词和语义匹配
        # 解析风格关键词
        style_params = self._parse_style_keywords(text_lower)

        # 如果关键词匹配没有结果，尝试使用语义匹配
        if not style_params and self.text_encoder is not None:
            style_params = self._semantic_style_match(text_lower)

        params = self._merge_params(params, style_params)

        # 解析具体参数调整
        adjustment_params = self._parse_adjustments(text_lower)
        params = self._merge_params(params, adjustment_params)

        # 解析主体/颜色语义，例如天空、肤色、草地、霓虹等
        semantic_color_params = self._parse_semantic_color_controls(text_lower)
        params = self._merge_params(params, semantic_color_params)

        # 应用程度修饰
        params = self._apply_intensity_modifiers(text_lower, params)

        # 始终尝试智能推断，以处理否定词和特定语境（覆盖之前的模糊匹配）
        params = self._intelligent_inference(original_text, params)

        print(f"[传统匹配] 参数: {self._format_params(params)}")

        return params

    def parse_async(
        self,
        text: str,
        on_success: Callable[[ColorGradingParams], None],
        on_error: Optional[Callable[[str], None]] = None,
        reference_params: Optional[ColorGradingParams] = None
    ):
        """
        异步解析自然语言描述生成调色参数（使用LLM）

        Args:
            text: 用户输入的自然语言描述
            on_success: 成功回调，接收ColorGradingParams
            on_error: 错误回调，接收错误信息
            reference_params: 参考图像的调色参数(用于"复刻"类请求)
        """
        original_text = text
        text_lower = text.lower().strip()

        print(f"[调色指令] 异步处理: {original_text}")

        # 检测是否是复刻请求
        if self._is_reference_request(text_lower) and reference_params:
            params = ColorGradingParams()
            params = self._merge_params(params, reference_params)
            on_success(params)
            return

        # 如果没有启用LLM，回退到同步传统匹配
        if not self.use_llm or not self.llm_analyzer:
            params = self._traditional_parse(text_lower, original_text)
            on_success(params)
            return

        # 使用异步LLM分析
        def handle_llm_success(result: Dict[str, Any]):
            """LLM分析成功回调"""
            try:
                if result.get("is_color_related", False):
                    print(f"[LLM分析] {result.get('reasoning', '无推理信息')}")

                    # 将LLM返回的参数合并到params
                    llm_params = result.get("parameters", {})
                    if llm_params:
                        params = ColorGradingParams()
                        params = self._merge_params(params, llm_params)
                        if not self._is_default_params(params):
                            print(f"[LLM分析] 参数: {self._format_params(params)}")
                            on_success(params)
                            return
                        print("[LLM分析] 返回了默认参数，回退到传统匹配")
                    else:
                        print("[LLM分析] 未返回参数，回退到传统匹配")
                else:
                    print(f"[LLM分析] 非调色指令: {result.get('reasoning', '')}，尝试传统匹配")

                params = self._traditional_parse(text_lower, original_text)
                on_success(params)

            except Exception as e:
                error_msg = f"[LLM分析] 结果处理失败: {e}"
                print(error_msg)
                params = self._traditional_parse(text_lower, original_text)
                on_success(params)

        def handle_llm_error(error_msg: str):
            """LLM分析失败回调"""
            print(f"[LLM分析] 失败: {error_msg}，回退到传统匹配")
            params = self._traditional_parse(text_lower, original_text)
            on_success(params)

        # 启动异步分析
        self.llm_analyzer.analyze_async(
            original_text,
            on_success=handle_llm_success,
            on_error=handle_llm_error
        )

    def _traditional_parse(self, text_lower: str, original_text: str) -> ColorGradingParams:
        """传统的关键词和语义匹配解析"""
        params = ColorGradingParams()

        # 解析风格关键词
        style_params = self._parse_style_keywords(text_lower)

        # 如果关键词匹配没有结果，尝试使用语义匹配
        if not style_params and self.text_encoder is not None:
            style_params = self._semantic_style_match(text_lower)

        params = self._merge_params(params, style_params)

        # 解析具体参数调整
        adjustment_params = self._parse_adjustments(text_lower)
        params = self._merge_params(params, adjustment_params)

        # 解析主体/颜色语义
        semantic_color_params = self._parse_semantic_color_controls(text_lower)
        params = self._merge_params(params, semantic_color_params)

        # 应用程度修饰
        params = self._apply_intensity_modifiers(text_lower, params)

        # 始终尝试智能推断
        params = self._intelligent_inference(original_text, params)

        print(f"[传统匹配] 参数: {self._format_params(params)}")

        return params

    def _format_params(self, params: ColorGradingParams) -> str:
        """格式化参数用于日志输出，动态显示修改过的参数"""
        default = ColorGradingParams()
        p_dict = params.to_dict()
        d_dict = default.to_dict()

        name_map = {
            "exposure": "曝光", "contrast": "对比度", "gamma": "Gamma",
            "temperature": "色温", "tint": "色调", "saturation": "饱和度",
            "vibrance": "自然饱和度", "highlights": "高光", "shadows": "阴影",
            "whites": "白色", "blacks": "黑色", "clarity": "清晰度",
            "texture": "纹理", "dehaze": "去雾", "blue_saturation": "蓝HSL",
            "shadow_saturation": "阴影色轮强度", "curve_shadows": "曲线阴影",
            "curve_highlights": "曲线高光", "bloom": "柔光", "vignette": "暗角",
            "grain": "颗粒", "sharpen": "锐化", "noise_reduction": "降噪"
        }
        for color_key, color_label in (
                ("red", "红色"), ("orange", "橙色"), ("yellow", "黄色"), ("green", "绿色"),
                ("aqua", "青色"), ("blue", "蓝色"), ("purple", "紫色"), ("magenta", "品红")):
            name_map.update({
                f"{color_key}_hue": f"{color_label}色相",
                f"{color_key}_saturation": f"{color_label}饱和",
                f"{color_key}_luminance": f"{color_label}明度",
            })

        changed = []
        for k, v in p_dict.items():
            d_v = d_dict.get(k, 0)
            if isinstance(v, (int, float)) and isinstance(d_v, (int, float)):
                if abs(v - d_v) > 0.01:
                    name = name_map.get(k, k)
                    if isinstance(v, float) and abs(v) <= 5:
                        changed.append(f"{name}={v:.2f}")
                    else:
                        changed.append(f"{name}={v:.0f}")
            else:
                if v != d_v:
                    name = name_map.get(k, k)
                    changed.append(f"{name}={v}")

        if not changed:
            return "参数保持默认"

        return ", ".join(changed[:15]) + ("..." if len(changed) > 15 else "")

    def _is_default_params(self, params: ColorGradingParams) -> bool:
        """检查参数是否为默认值"""
        default = ColorGradingParams()
        params_dict = params.to_dict()
        default_dict = default.to_dict()

        for key, value in params_dict.items():
            default_value = default_dict.get(key)
            if isinstance(value, (int, float)) and isinstance(default_value, (int, float)):
                if abs(value - default_value) > 0.01:
                    return False
            elif value != default_value:
                return False
        return True

    def _semantic_style_match(self, text: str) -> Dict[str, Any]:
        """使用语义相似度匹配风格"""
        if self.text_encoder is None:
            return {}

        try:
            # 编码输入文本
            text_embedding = self.text_encoder.encode([text])

            # 编码所有风格关键词
            style_names = list(self.COLOR_KEYWORDS.keys())
            style_embeddings = self.text_encoder.encode(style_names)

            # 计算相似度
            similarities = np.dot(text_embedding, style_embeddings.T)[0]

            # 找到最相似的风格
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # 相似度阈值（至少0.3才算匹配）
            if best_similarity > 0.3:
                best_style = style_names[best_idx]
                print(f"语义匹配: '{text}' -> '{best_style}' (相似度: {best_similarity:.2f})")
                return self.COLOR_KEYWORDS[best_style].copy()

        except Exception as e:
            print(f"语义匹配失败: {e}")

        return {}

    def _intelligent_inference(self, text: str, params: ColorGradingParams) -> ColorGradingParams:
        """
        智能推断调色参数
        支持否定词检测，避免误解用户意图
        """
        text_lower = text.lower()

        # 否定词列表
        negation_words = ['不', '别', '勿', '不要', '不用', '无需', '别太', '不能太', '避免']

        # 检测是否有否定词（检查否定词周围的上下文）
        def has_negation_near(keyword: str, text: str) -> bool:
            """检测关键词附近是否有否定词"""
            pos = text.find(keyword)
            if pos == -1:
                return False
            # 检查关键词前5个字符内是否有否定词
            start = max(0, pos - 5)
            context = text[start:pos + len(keyword) + 2]
            return any(neg in context for neg in negation_words)

        # 亮度相关词汇
        brightness_pos = ['亮', '明亮', '提亮', '光亮']
        brightness_neg = ['暗', '阴暗', '压暗', '昏暗']

        if any(w in text_lower for w in brightness_pos):
            # 检查是否有否定
            is_negated = any(has_negation_near(w, text_lower) for w in brightness_pos if w in text_lower)
            params.exposure = -0.2 if is_negated else 0.3

        if any(w in text_lower for w in brightness_neg):
            is_negated = any(has_negation_near(w, text_lower) for w in brightness_neg if w in text_lower)
            params.exposure = 0.3 if is_negated else -0.2

        # 色温相关词汇
        cool_words = ['冷', '冷色', '清冷', '冰冷', '蓝调']
        warm_words = ['暖', '暖色', '温暖', '黄', '金黄']

        if any(w in text_lower for w in cool_words):
            is_negated = any(has_negation_near(w, text_lower) for w in cool_words if w in text_lower)
            params.temperature = 25 if is_negated else -25

        if any(w in text_lower for w in warm_words):
            is_negated = any(has_negation_near(w, text_lower) for w in warm_words if w in text_lower)
            params.temperature = -25 if is_negated else 25

        # 对比度相关
        high_contrast = ['对比', '层次', '立体']
        low_contrast = ['柔和', '柔软', '温柔']

        if any(w in text_lower for w in high_contrast):
            is_negated = any(has_negation_near(w, text_lower) for w in high_contrast if w in text_lower)
            params.contrast = 0.9 if is_negated else 1.2

        if any(w in text_lower for w in low_contrast):
            is_negated = any(has_negation_near(w, text_lower) for w in low_contrast if w in text_lower)
            params.contrast = 1.2 if is_negated else 0.9

        # 饱和度相关
        vivid_words = ['鲜艳', '色彩', '彩色', '艳丽']
        dull_words = ['淡', '素', '灰']

        if any(w in text_lower for w in vivid_words):
            is_negated = any(has_negation_near(w, text_lower) for w in vivid_words if w in text_lower)
            if is_negated:
                params.saturation = 0.7
            else:
                params.saturation = 1.3
                params.vibrance = 20

        if any(w in text_lower for w in dull_words):
            is_negated = any(has_negation_near(w, text_lower) for w in dull_words if w in text_lower)
            if is_negated:
                params.saturation = 1.3
                params.vibrance = 20
            else:
                params.saturation = 0.7

        # 清晰度相关
        sharp_words = ['清晰', '锐利', '锐化']
        soft_words = ['朦胧', '模糊', '梦幻']

        if any(w in text_lower for w in sharp_words):
            is_negated = any(has_negation_near(w, text_lower) for w in sharp_words if w in text_lower)
            params.clarity = -20 if is_negated else 25

        if any(w in text_lower for w in soft_words):
            is_negated = any(has_negation_near(w, text_lower) for w in soft_words if w in text_lower)
            if is_negated:
                params.clarity = 25
            else:
                params.clarity = -20
                params.fade = 0.1

        return params

    def _is_reference_request(self, text: str) -> bool:
        """检测是否是参考/复刻类请求"""
        reference_keywords = ['复刻', '参考', '像', '类似', '风格', '那种', '同样']
        return any(kw in text for kw in reference_keywords)

    def _parse_style_keywords(self, text: str) -> Dict[str, Any]:
        """解析风格关键词"""
        params = {}

        for keyword, values in self.COLOR_KEYWORDS.items():
            if keyword in text:
                for key, value in values.items():
                    if key not in params:
                        params[key] = value
                    else:
                        # 混合多个风格的参数
                        if isinstance(value, (int, float)):
                            params[key] = (params[key] + value) / 2

        return params

    def _parse_adjustments(self, text: str) -> Dict[str, Any]:
        """解析具体参数调整指令"""
        params = {}

        # 检测动作+参数的组合
        for action, multiplier in self.ACTION_KEYWORDS.items():
            if action in text:
                for param_keyword, param_name in self.PARAM_KEYWORDS.items():
                    if param_keyword in text:
                        # 根据参数类型设置调整值
                        if param_name == 'exposure':
                            params[param_name] = 0.3 * multiplier
                        elif param_name == 'brightness':
                            params[param_name] = 0.14 * multiplier
                        elif param_name == 'contrast':
                            params[param_name] = 1.0 + 0.2 * multiplier
                        elif param_name == 'gamma':
                            params[param_name] = 1.0 + 0.22 * multiplier
                        elif param_name == 'saturation':
                            params[param_name] = 1.0 + 0.2 * multiplier
                        elif param_name in ['temperature', 'tint', 'clarity', 'dehaze',
                                           'highlights', 'shadows', 'whites', 'blacks', 'vibrance',
                                           'texture', 'midtone_detail', 'curve_shadows',
                                           'curve_darks', 'curve_lights', 'curve_highlights']:
                            params[param_name] = 25 * multiplier
                        elif param_name in ['sharpen', 'bloom', 'vignette']:
                            params[param_name] = 20 * abs(multiplier) if multiplier > 0 else 0
                        elif param_name == 'noise_reduction':
                            params[param_name] = 25 * abs(multiplier)
                        elif param_name == 'grain':
                            if multiplier < 0:
                                params['grain'] = 0
                                params['noise_reduction'] = 20 * abs(multiplier)
                            else:
                                params[param_name] = 20 * multiplier
                        elif param_name.endswith('_saturation'):
                            params[param_name] = 25 * multiplier

        return params

    def _parse_semantic_color_controls(self, text: str) -> Dict[str, Any]:
        """解析更接近摄影调色语言的主体/颜色语义。"""
        params: Dict[str, Any] = {}

        def has_any(words: List[str]) -> bool:
            return any(word in text for word in words)

        def merge(values: Dict[str, Any]):
            for key, value in values.items():
                if key not in params:
                    params[key] = value
                elif isinstance(value, (int, float)) and isinstance(params[key], (int, float)):
                    params[key] = (params[key] + value) / 2
                else:
                    params[key] = value

        strong = has_any(['更', '加强', '增强', '浓', '鲜艳', '通透', '明显'])
        soft = has_any(['稍微', '一点', '轻微', '淡一点', '别太', '不要太'])
        scale = 1.35 if strong else 0.65 if soft else 1.0

        if has_any(['天空', '蓝天', '云天']):
            merge({
                'blue_saturation': 24 * scale,
                'blue_luminance': -6 * scale,
                'aqua_saturation': 12 * scale,
                'dehaze': max(params.get('dehaze', 0), 12 * scale),
            })
            if has_any(['清澈', '通透', '干净']):
                merge({'clarity': 10 * scale, 'dehaze': 22 * scale})

        if has_any(['海', '海水', '大海', '湖水', '湖面']):
            merge({
                'aqua_saturation': 22 * scale,
                'blue_saturation': 18 * scale,
                'aqua_luminance': 6 * scale,
                'temperature': -8 * scale,
                'dehaze': 12 * scale,
            })

        if has_any(['草地', '草坪', '树叶', '森林', '树林', '植被', '绿植', '绿色', '绿意']):
            merge({
                'green_saturation': 22 * scale,
                'green_hue': -6 * scale,
                'green_luminance': -4 * scale,
                'yellow_saturation': -8 * scale,
            })
            if has_any(['深绿', '深绿色', '墨绿', '暗绿']):
                merge({
                    'exposure': -0.08 * scale,
                    'gamma': 0.92,
                    'green_saturation': 32 * scale,
                    'green_luminance': -14 * scale,
                    'yellow_saturation': -16 * scale,
                    'curve_darks': -8 * scale,
                    'midtone_hue': 132,
                    'midtone_saturation': 18 * scale,
                })
            if has_any(['森系', '清冷']):
                merge({'temperature': -8 * scale, 'tint': -10 * scale})

        if has_any(['肤色', '皮肤', '人像', '脸', '面部']):
            merge({
                'orange_luminance': 10 * scale,
                'orange_saturation': 6 * scale,
                'red_saturation': -8 * scale,
                'texture': -12 * scale,
                'clarity': -6 * scale,
            })
            if has_any(['别太红', '不要太红', '不红', '去红', '压红']):
                merge({'red_saturation': -24 * scale, 'red_hue': 4 * scale})
            if has_any(['白皙', '透亮', '干净']):
                merge({'orange_luminance': 16 * scale, 'noise_reduction': 14 * scale})

        if has_any(['夕阳', '日落', '晚霞', '金色', '金黄', '暖阳']):
            merge({
                'temperature': 24 * scale,
                'orange_saturation': 16 * scale,
                'yellow_saturation': 12 * scale,
                'highlight_hue': 38,
                'highlight_saturation': 24 * scale,
            })

        if has_any(['霓虹', '赛博', '夜店', '荧光']):
            merge({
                'contrast': 1.26,
                'vibrance': 32 * scale,
                'blue_saturation': 18 * scale,
                'purple_saturation': 20 * scale,
                'magenta_saturation': 24 * scale,
                'bloom': 22 * scale,
                'shadow_hue': 235,
                'shadow_saturation': 22 * scale,
                'highlight_hue': 305,
                'highlight_saturation': 26 * scale,
            })

        if has_any(['胶片', '复古', '怀旧', '老照片']):
            merge({
                'fade': 0.14 * scale,
                'grain': 18 * scale,
                'curve_shadows': 12 * scale,
                'curve_highlights': -6 * scale,
                'saturation': 0.88,
            })

        if has_any(['柔雾', '朦胧', '梦幻', '柔焦']):
            merge({
                'clarity': -20 * scale,
                'texture': -16 * scale,
                'dehaze': -18 * scale,
                'bloom': 18 * scale,
            })

        if has_any(['清透', '通透', '干净', '透亮']):
            merge({
                'dehaze': max(params.get('dehaze', 0), 20 * scale),
                'clarity': max(params.get('clarity', 0), 12 * scale),
                'midtone_detail': max(params.get('midtone_detail', 0), 10 * scale),
                'noise_reduction': max(params.get('noise_reduction', 0), 8 * scale),
            })

        if has_any(['青橙', '青蓝橙', '电影蓝橙']):
            merge({
                'contrast': 1.22,
                'saturation': 0.9,
                'shadow_hue': 195,
                'shadow_saturation': 28 * scale,
                'highlight_hue': 38,
                'highlight_saturation': 24 * scale,
                'curve_darks': -10 * scale,
                'curve_lights': 8 * scale,
            })

        return params

    def _apply_intensity_modifiers(self, text: str, params: ColorGradingParams) -> ColorGradingParams:
        """应用程度修饰词"""
        intensity = 1.0

        for modifier, value in self.INTENSITY_MODIFIERS.items():
            if modifier in text:
                intensity = value
                break

        if intensity != 1.0:
            # 对数值类参数应用强度调整
            params.exposure *= intensity
            params.brightness *= intensity
            params.contrast = 1.0 + (params.contrast - 1.0) * intensity
            params.gamma = 1.0 + (params.gamma - 1.0) * intensity
            params.saturation = 1.0 + (params.saturation - 1.0) * intensity
            params.temperature *= intensity
            params.tint *= intensity
            params.vibrance *= intensity
            params.clarity *= intensity
            params.texture *= intensity
            params.midtone_detail *= intensity
            params.sharpen *= intensity
            params.noise_reduction *= intensity
            params.dehaze *= intensity
            params.bloom *= intensity
            params.highlights *= intensity
            params.shadows *= intensity
            params.whites *= intensity
            params.blacks *= intensity
            params.vignette *= intensity
            params.grain *= intensity
            params.fade *= intensity
            params.curve_shadows *= intensity
            params.curve_darks *= intensity
            params.curve_lights *= intensity
            params.curve_highlights *= intensity
            params.shadow_saturation *= intensity
            params.midtone_saturation *= intensity
            params.highlight_saturation *= intensity

            for color_name in [
                    'red', 'orange', 'yellow', 'green',
                    'aqua', 'blue', 'purple', 'magenta']:
                for suffix in ['hue', 'saturation', 'luminance']:
                    attr = f'{color_name}_{suffix}'
                    setattr(params, attr, getattr(params, attr) * intensity)

        return params

    def _merge_params(self, base: ColorGradingParams,
                      updates: [Dict[str, Any], ColorGradingParams]) -> ColorGradingParams:
        """合并参数"""
        if isinstance(updates, ColorGradingParams):
            updates = updates.to_dict()
        else:
            updates = ColorGradingParams.normalize_dict(updates)

        base_dict = base.to_dict()
        for key, value in updates.items():
            if key in base_dict and value is not None:
                base_dict[key] = value

        return ColorGradingParams.from_dict(base_dict)

    def get_style_suggestions(self, text: str) -> List[str]:
        """根据输入文本获取风格建议"""
        suggestions = []

        # 基于关键词匹配
        for keyword in self.COLOR_KEYWORDS.keys():
            if any(char in text for char in keyword):
                suggestions.append(keyword)

        # 如果有文本编码器,使用语义相似度
        if self.text_encoder and len(suggestions) < 5:
            try:
                text_embedding = self.text_encoder.encode([text])
                style_embeddings = self.text_encoder.encode(list(self.COLOR_KEYWORDS.keys()))

                similarities = np.dot(text_embedding, style_embeddings.T)[0]
                top_indices = np.argsort(similarities)[-5:][::-1]

                style_names = list(self.COLOR_KEYWORDS.keys())
                for idx in top_indices:
                    if style_names[idx] not in suggestions:
                        suggestions.append(style_names[idx])
            except Exception:
                pass

        return suggestions[:5]
