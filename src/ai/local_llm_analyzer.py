"""
本地大模型调色分析器
使用transformers直接加载轻量化模型（如Qwen2.5-1.5B-Instruct）
"""
import json
import torch
from typing import Dict, Any, Optional
from pathlib import Path


class LocalLLMColorAnalyzer:
    """本地大模型调色分析器"""

    SYSTEM_PROMPT = """你是一个专业的色彩调色专家。你的任务是理解用户的一句话描述，并将其转换为具体的调色参数。

只返回需要改变的参数，不要返回默认值。所有数值都要温和、可叠加，除非用户明确要求极端风格。

可用参数：
1. 基础影调
- exposure: 曝光[-2,2]，正值变亮，负值变暗
- brightness: 亮度[-1,1]，整体加减亮度
- contrast: 对比度[0.5,2]，默认1.0
- gamma: 中间调亮度[0.25,3]，默认1.0，>1提亮中间调，<1压暗中间调
- highlights/shadows/whites/blacks: 高光/阴影/白色/黑色[-100,100]
- curve_shadows/curve_darks/curve_lights/curve_highlights: 参数曲线[-100,100]

2. 全局色彩
- temperature: 色温[-100,100]，负值偏蓝，正值偏黄
- tint: 色调[-100,100]，负值偏绿，正值偏品红
- saturation: 全局饱和度[0,2]，默认1.0
- vibrance: 自然饱和度[-100,100]
- hue_shift: 全局色相偏移[-180,180]
- red_balance/green_balance/blue_balance: RGB通道平衡[-100,100]

3. 分色 HSL，适合“天空更蓝、草地更绿、肤色更通透”
- red/orange/yellow/green/aqua/blue/purple/magenta + _hue: 单色相偏移[-60,60]
- red/orange/yellow/green/aqua/blue/purple/magenta + _saturation: 单色饱和度[-100,100]
- red/orange/yellow/green/aqua/blue/purple/magenta + _luminance: 单色明度[-100,100]
- 也可用嵌套 hsl: {"blue": {"saturation": 25, "luminance": -10}}

4. 三路色轮，色相单位为0-360度，强度为[0,100]
- shadow_hue/shadow_saturation: 阴影染色
- midtone_hue/midtone_saturation: 中间调染色
- highlight_hue/highlight_saturation: 高光染色
- 也可用嵌套 color_wheels: {"shadows": {"hue": 190, "saturation": 25}}

5. 质感与镜头效果
- clarity: 清晰度/局部对比[-100,100]
- texture: 纹理[-100,100]，负值柔化皮肤，正值增强细节
- midtone_detail: 中间调细节[-100,100]
- sharpen: 锐化[0,100]
- noise_reduction: 降噪[0,100]
- dehaze: 去雾[-100,100]，负值增加柔雾
- bloom: 高光柔光[0,100]
- vignette: 暗角[0,100]
- grain: 颗粒[0,100]
- fade: 褪色/抬黑[0,1]
- tone_curve: 自定义曲线点，例如 [[0,18],[64,58],[128,132],[255,246]]

6. 专业通道校正，可选
- cdl_slope/cdl_offset/cdl_power: RGB三元数组
- cdl_saturation: CDL饱和度，默认1.0

语义映射建议：
- “电影感/青橙/大片”：降低全局饱和，提升对比，阴影偏青蓝，高光偏橙金，可加轻微S曲线和暗角。
- “日系/空气感/清新”：提曝光，降低对比和饱和，提阴影，蓝/青略亮，纹理柔和。
- “胶片/复古/Kodak/颗粒”：轻微暖色、褪色、颗粒、抬黑，曲线压高光或提暗部。
- “赛博朋克/霓虹”：提高对比和自然饱和，阴影偏蓝紫，高光偏品红/青，bloom略高。
- “人像/肤色好看”：橙色明度略升、橙色饱和温和，降低纹理或清晰度，避免肤色过红。
- “天空/海水”：优先使用 blue/aqua 的 HSL，而不是全局色温。
- “森林/草地”：优先使用 green/yellow 的 HSL，可轻微压黄提绿。
- “低调/暗黑/情绪”：降曝光或gamma，压阴影/黑色，提升对比，暗角。
- “通透/干净”：dehaze、clarity、midtone_detail略增，噪点少，白色和高光谨慎提升。

重要规则：
1. 仔细分析用户描述的含义、主体和色彩特征。
2. 如果是具体事物（如“太阳”“大海”“胶片海报”），提取典型色相、影调和质感。
3. 如果描述与调色无关（如单纯菜名、闲聊、文件操作），返回 is_color_related: false。
4. 返回JSON格式，必须包含字段：is_color_related, reasoning, parameters。

示例1：
输入：“青橙电影感，暗部冷一点，高光像夕阳”
输出：{"is_color_related": true, "reasoning": "青橙电影感需要较强影调对比、暗部青蓝、高光橙金，同时压低全局饱和避免艳俗", "parameters": {"contrast": 1.22, "saturation": 0.88, "curve_shadows": -12, "curve_highlights": 10, "color_wheels": {"shadows": {"hue": 195, "saturation": 28}, "highlights": {"hue": 38, "saturation": 24}}, "vignette": 12}}

示例2：
输入：“让天空更蓝更通透，但人物肤色别太红”
输出：{"is_color_related": true, "reasoning": "天空应使用蓝色和青色HSL定向增强，同时降低红色饱和以保护肤色", "parameters": {"hsl": {"blue": {"saturation": 30, "luminance": -8}, "aqua": {"saturation": 18, "luminance": 8}, "red": {"saturation": -12}, "orange": {"luminance": 8}}, "dehaze": 18, "clarity": 8}}

示例3：
输入：“鱼香肉丝”
输出：{"is_color_related": false, "reasoning": "这是一道菜名，不是调色指令", "parameters": {}}

请严格按照JSON格式返回，不要包含其他文字。"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto",
                 quantization_config: Optional[Dict[str, Any]] = None,
                 max_memory: Optional[Dict[str, str]] = None,
                 offload_folder: Optional[str] = None,
                 trust_remote_code: bool = False):
        """
        初始化本地模型分析器

        Args:
            model_name: 模型名称或路径
            device: 设备，"auto"/"cuda"/"cpu"
            quantization_config: 量化配置
                - enabled: bool, 是否启用量化
                - bits: int, 量化位数 (4/8)
                - compute_dtype: str, 计算精度 ("float16"/"bfloat16")
            max_memory: 最大内存限制 {"gpu": "6GB", "cpu": "8GB"}
            offload_folder: CPU卸载文件夹
            trust_remote_code: 是否允许执行模型仓库中的自定义代码
        """
        self.model_name = model_name
        self.device = device
        self.quantization_config = quantization_config or {}
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载模型（支持量化和内存管理）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            print(f"正在加载本地模型: {self.model_name}")

            # 判断设备
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                    print("✓ 检测到CUDA，使用GPU加速")
                else:
                    device = "cpu"
                    print("✓ 使用CPU模式")
            else:
                device = self.device

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            # 构建加载参数
            load_kwargs = {
                "trust_remote_code": self.trust_remote_code,
            }

            # 量化配置
            if self.quantization_config.get("enabled", False) and device != "cpu":
                try:
                    bits = self.quantization_config.get("bits", 4)
                    compute_dtype = self.quantization_config.get("compute_dtype", "float16")

                    compute_dtype_map = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32
                    }

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=(bits == 4),
                        load_in_8bit=(bits == 8),
                        bnb_4bit_compute_dtype=compute_dtype_map.get(compute_dtype, torch.float16),
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    load_kwargs["quantization_config"] = quantization_config
                    print(f"✓ 启用 {bits}-bit 量化 (节省内存 ~75%)")
                except Exception as e:
                    print(f"⚠ 量化失败，使用标准加载: {e}")
                    print("  提示: 安装 bitsandbytes 以启用量化: pip install bitsandbytes")

            # 内存管理
            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory
                print(f"✓ 内存限制: {self.max_memory}")

            if self.offload_folder and device != "cpu":
                load_kwargs["offload_folder"] = self.offload_folder
                print(f"✓ CPU卸载目录: {self.offload_folder}")

            # 设备映射
            if device != "cpu":
                load_kwargs["device_map"] = "auto"

            # CPU模式下使用更少的内存
            if device == "cpu":
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["low_cpu_mem_usage"] = True
            elif "quantization_config" not in load_kwargs:
                # GPU模式下未量化时使用半精度
                load_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            if device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()  # 设置为评估模式

            # 计算模型大小
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9

            print(f"✓ 模型加载成功: {self.model_name}")
            print(f"  设备: {device}")
            print(f"  参数量: {param_count:.2f}B")

            # 估算内存占用
            if self.quantization_config.get("enabled"):
                bits = self.quantization_config.get("bits", 4)
                memory_gb = param_count * bits / 8
                print(f"  估算显存: ~{memory_gb:.1f}GB ({bits}-bit)")
            else:
                dtype = load_kwargs.get("torch_dtype", torch.float32)
                bytes_per_param = 2 if dtype == torch.float16 else 4
                memory_gb = param_count * bytes_per_param
                print(f"  估算显存: ~{memory_gb:.1f}GB")

        except ImportError as e:
            if "bitsandbytes" in str(e):
                print("⚠ 量化需要 bitsandbytes 库")
                print("  安装命令: pip install bitsandbytes")
            raise ImportError("需要安装transformers库: pip install transformers torch")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise

    def analyze(self, description: str) -> Dict[str, Any]:
        """
        分析用户描述并返回调色参数

        Args:
            description: 用户的描述文本

        Returns:
            包含以下字段的字典：
            - is_color_related: bool，是否与色彩相关
            - reasoning: str，分析推理过程
            - parameters: dict，调色参数
        """
        if self.model is None or self.tokenizer is None:
            return {
                "is_color_related": False,
                "reasoning": "模型未加载",
                "parameters": {}
            }

        try:
            # 构建对话
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f'请分析以下描述并生成调色参数：\n\n"{description}"\n\n请返回JSON格式的分析结果。'}
            ]

            # 使用apply_chat_template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 编码输入
            inputs = self.tokenizer([text], return_tensors="pt")

            # 移动到正确的设备
            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            response = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )

            # 提取JSON
            result_text = self._extract_json(response)

            # 解析JSON
            result = json.loads(result_text)

            # 验证结果格式
            if not isinstance(result, dict):
                raise ValueError("返回结果不是字典格式")

            if "is_color_related" not in result:
                result["is_color_related"] = True

            if "parameters" not in result:
                result["parameters"] = {}

            if "reasoning" not in result:
                result["reasoning"] = "AI分析"

            return result

        except Exception as e:
            print(f"本地模型分析失败: {e}")
            return {
                "is_color_related": False,
                "reasoning": f"分析失败: {str(e)}",
                "parameters": {}
            }

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON部分"""
        # 尝试找到JSON的开始和结束
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            return text[start:end+1]

        return text

    def unload(self):
        """卸载模型以释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ 模型已卸载，内存已释放")

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况（GB）"""
        memory_info = {}

        if self.model is not None and torch.cuda.is_available():
            # GPU内存
            memory_info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3
            memory_info["gpu_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3

        # CPU内存（需要psutil）
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_rss"] = process.memory_info().rss / 1024**3
        except ImportError:
            pass

        return memory_info

    def __del__(self):
        """析构函数：确保资源释放"""
        self.unload()
