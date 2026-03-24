"""
自然语言调色参数解析器
将用户的自然语言描述转换为具体的调色参数
"""
import re
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ColorGradingParams:
    """调色参数数据类"""
    # 基础调整
    exposure: float = 0.0          # 曝光 [-2, 2]
    contrast: float = 1.0          # 对比度 [0.5, 2]
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

    # 分离色调
    split_tone_shadows: List[int] = field(default_factory=lambda: [0, 0, 0])
    split_tone_highlights: List[int] = field(default_factory=lambda: [255, 255, 255])
    split_tone_balance: float = 0.0  # [-100, 100]

    # 效果
    clarity: float = 0.0           # 清晰度 [-100, 100]
    dehaze: float = 0.0            # 去雾 [-100, 100]
    vignette: float = 0.0          # 暗角 [0, 100]
    grain: float = 0.0             # 颗粒 [0, 100]
    fade: float = 0.0              # 褪色 [0, 1]

    # 曲线 (可选的自定义曲线点)
    tone_curve: Optional[List[Tuple[int, int]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'exposure': self.exposure,
            'contrast': self.contrast,
            'highlights': self.highlights,
            'shadows': self.shadows,
            'whites': self.whites,
            'blacks': self.blacks,
            'temperature': self.temperature,
            'tint': self.tint,
            'vibrance': self.vibrance,
            'saturation': self.saturation,
            'hue_shift': self.hue_shift,
            'split_tone_shadows': self.split_tone_shadows,
            'split_tone_highlights': self.split_tone_highlights,
            'split_tone_balance': self.split_tone_balance,
            'clarity': self.clarity,
            'dehaze': self.dehaze,
            'vignette': self.vignette,
            'grain': self.grain,
            'fade': self.fade,
            'tone_curve': self.tone_curve
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColorGradingParams':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


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
        '橙调': {'temperature': 30, 'hue_shift': 15},
        '绿调': {'tint': -20, 'hue_shift': 60},
        '粉调': {'tint': 25, 'saturation': 0.95},
        '紫调': {'hue_shift': -30, 'tint': 15},

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

        # 情绪关键词
        '明亮': {'exposure': 0.2, 'highlights': 15},
        '阴暗': {'exposure': -0.15, 'shadows': -20},
        '柔和': {'contrast': 0.9, 'clarity': -15},
        '锐利': {'contrast': 1.15, 'clarity': 30},
        '梦幻': {'contrast': 0.85, 'saturation': 0.9, 'fade': 0.2, 'clarity': -20},
        '通透': {'dehaze': 25, 'clarity': 20, 'vibrance': 10},
        '高级感': {'contrast': 1.1, 'saturation': 0.85, 'clarity': 15},
        '质感': {'clarity': 25, 'contrast': 1.1},
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
        '对比度': 'contrast',
        '对比': 'contrast',
        '饱和度': 'saturation',
        '饱和': 'saturation',
        '色温': 'temperature',
        '色调': 'tint',
        '清晰度': 'clarity',
        '锐度': 'clarity',
        '去雾': 'dehaze',
        '暗角': 'vignette',
        '颗粒': 'grain',
        '噪点': 'grain',
        '高光': 'highlights',
        '阴影': 'shadows',
        '白色': 'whites',
        '黑色': 'blacks',
        '自然饱和度': 'vibrance',
        '鲜艳度': 'vibrance',
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
                device=device
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
                        print(f"[LLM分析] 参数: {self._format_params(params)}")
                        return params
                else:
                    print(f"[LLM分析] 非调色指令: {result.get('reasoning', '')}")
                    # 返回默认参数
                    return params

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
                        print(f"[LLM分析] 参数: {self._format_params(params)}")
                        on_success(params)
                    else:
                        on_success(ColorGradingParams())
                else:
                    print(f"[LLM分析] 非调色指令: {result.get('reasoning', '')}")
                    # 返回默认参数
                    on_success(ColorGradingParams())

            except Exception as e:
                error_msg = f"[LLM分析] 结果处理失败: {e}"
                print(error_msg)
                if on_error:
                    on_error(error_msg)
                else:
                    # 回退到传统匹配
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

        # 应用程度修饰
        params = self._apply_intensity_modifiers(text_lower, params)

        # 始终尝试智能推断
        params = self._intelligent_inference(original_text, params)

        print(f"[传统匹配] 参数: {self._format_params(params)}")

        return params

    def _format_params(self, params: ColorGradingParams) -> str:
        """格式化参数用于日志输出"""
        return (f"曝光={params.exposure:.2f}, 对比度={params.contrast:.2f}, "
                f"色温={params.temperature:.0f}, 饱和度={params.saturation:.2f}")
    
    def _is_default_params(self, params: ColorGradingParams) -> bool:
        """检查参数是否为默认值"""
        default = ColorGradingParams()
        return (
            params.exposure == default.exposure and
            params.contrast == default.contrast and
            params.temperature == default.temperature and
            params.saturation == default.saturation and
            params.vibrance == default.vibrance
        )
    
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
        cool_words = ['冷', '冷色', '蓝', '清冷', '冰冷']
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
                        elif param_name == 'contrast':
                            params[param_name] = 1.0 + 0.2 * multiplier
                        elif param_name == 'saturation':
                            params[param_name] = 1.0 + 0.2 * multiplier
                        elif param_name in ['temperature', 'tint', 'clarity', 'dehaze',
                                           'highlights', 'shadows', 'whites', 'blacks', 'vibrance']:
                            params[param_name] = 25 * multiplier
                        elif param_name in ['vignette', 'grain']:
                            params[param_name] = 20 * abs(multiplier)

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
            params.contrast = 1.0 + (params.contrast - 1.0) * intensity
            params.saturation = 1.0 + (params.saturation - 1.0) * intensity
            params.temperature *= intensity
            params.tint *= intensity
            params.vibrance *= intensity
            params.clarity *= intensity
            params.dehaze *= intensity
            params.highlights *= intensity
            params.shadows *= intensity

        return params

    def _merge_params(self, base: ColorGradingParams,
                      updates: [Dict[str, Any], ColorGradingParams]) -> ColorGradingParams:
        """合并参数"""
        if isinstance(updates, ColorGradingParams):
            updates = updates.to_dict()

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
