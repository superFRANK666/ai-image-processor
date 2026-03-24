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

    SYSTEM_PROMPT = """你是一个专业的色彩调色专家。你的任务是理解用户的描述，并将其转换为具体的调色参数。

参数说明：
- exposure: 曝光，范围[-2, 2]，正值变亮，负值变暗
- contrast: 对比度，范围[0.5, 2]，默认1.0
- temperature: 色温，范围[-100, 100]，负值偏蓝（冷色），正值偏黄（暖色）
- tint: 色调，范围[-100, 100]，负值偏绿，正值偏品红
- saturation: 饱和度，范围[0, 2]，默认1.0
- vibrance: 自然饱和度，范围[-100, 100]
- clarity: 清晰度，范围[-100, 100]
- highlights: 高光，范围[-100, 100]
- shadows: 阴影，范围[-100, 100]
- hue_shift: 色相偏移，范围[-180, 180]

重要规则：
1. 仔细分析用户描述的含义和色彩特征
2. 如果是具体事物（如"太阳"、"大海"），分析其典型色彩特征
3. 如果是抽象概念（如"温暖"、"清新"），转换为对应的色彩调整
4. 如果描述与色彩无关（如"鱼香肉丝"这种菜名），返回 is_color_related: false
5. 返回JSON格式，必须包含字段：is_color_related, reasoning, parameters

示例1：
输入："太阳色"
输出：{"is_color_related": true, "reasoning": "太阳呈现金黄色、橙色的暖色调，色温高，饱和度高", "parameters": {"temperature": 45, "saturation": 1.25, "vibrance": 20, "exposure": 0.15}}

示例2：
输入："大海"
输出：{"is_color_related": true, "reasoning": "大海呈现蓝色、青色的冷色调，饱和度中等偏高", "parameters": {"temperature": -20, "saturation": 1.15, "vibrance": 15, "tint": -5}}

示例3：
输入："鱼香肉丝"
输出：{"is_color_related": false, "reasoning": "这是一道菜名，不是色彩调整指令", "parameters": {}}

请严格按照JSON格式返回，不要包含其他文字。"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto",
                 quantization_config: Optional[Dict[str, Any]] = None,
                 max_memory: Optional[Dict[str, str]] = None,
                 offload_folder: Optional[str] = None):
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
        """
        self.model_name = model_name
        self.device = device
        self.quantization_config = quantization_config or {}
        self.max_memory = max_memory
        self.offload_folder = offload_folder
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
                trust_remote_code=True
            )

            # 构建加载参数
            load_kwargs = {
                "trust_remote_code": True,
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
