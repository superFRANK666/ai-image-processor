"""
配置加载工具
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载本地大模型配置

    Args:
        config_path: 配置文件路径，默认为项目根目录的llm_config.json

    Returns:
        配置字典，包含 enabled, model_name, device, quantization, max_memory 等字段
    """
    if config_path is None:
        # 默认路径
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "llm_config.json"
    else:
        config_path = Path(config_path)

    # 如果配置文件不存在，返回禁用状态
    if not config_path.exists():
        print(f"LLM配置文件不存在: {config_path}")
        print("使用传统关键词匹配模式")
        return {"enabled": False}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证配置格式
        if not isinstance(config, dict):
            print("配置文件格式错误")
            return {"enabled": False}

        enabled = config.get("enabled", False)

        if not enabled:
            print("LLM功能已禁用")
            return {"enabled": False}

        model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        device = config.get("device", "auto")
        quantization = config.get("quantization", {})
        max_memory = config.get("max_memory", None)
        offload_folder = config.get("offload_folder", None)

        info_parts = [f"model={model_name}", f"device={device}"]
        if quantization.get("enabled"):
            info_parts.append(f"quant={quantization.get('bits', 4)}-bit")
        if max_memory:
            info_parts.append(f"max_mem={max_memory}")

        print(f"✓ 加载本地LLM配置: {', '.join(info_parts)}")

        return {
            "enabled": True,
            "model_name": model_name,
            "device": device,
            "quantization": quantization,
            "max_memory": max_memory,
            "offload_folder": offload_folder
        }

    except json.JSONDecodeError as e:
        print(f"配置文件JSON解析错误: {e}")
        return {"enabled": False}
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {"enabled": False}
