"""
配置加载工具
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def _safe_print(message: str):
    """打印状态信息，避免控制台编码问题影响配置加载。"""
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_message)


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
        _safe_print(f"LLM配置文件不存在: {config_path}")
        _safe_print("使用传统关键词匹配模式")
        return {"enabled": False}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证配置格式
        if not isinstance(config, dict):
            _safe_print("配置文件格式错误")
            return {"enabled": False}

        enabled = config.get("enabled", False)

        if not enabled:
            _safe_print("LLM功能已禁用")
            return {"enabled": False}

        model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
        device = config.get("device", "auto")
        quantization = config.get("quantization", {})
        if not isinstance(quantization, dict):
            _safe_print("量化配置格式错误，已忽略")
            quantization = {}

        max_memory = config.get("max_memory", None)
        if max_memory is not None and not isinstance(max_memory, dict):
            _safe_print("max_memory 配置格式错误，已忽略")
            max_memory = None
        offload_folder = config.get("offload_folder", None)
        trust_remote_code = bool(config.get("trust_remote_code", False))

        info_parts = [f"model={model_name}", f"device={device}"]
        if quantization.get("enabled"):
            info_parts.append(f"quant={quantization.get('bits', 4)}-bit")
        if max_memory:
            info_parts.append(f"max_mem={max_memory}")
        if trust_remote_code:
            info_parts.append("trust_remote_code=true")

        _safe_print(f"[OK] 加载本地LLM配置: {', '.join(info_parts)}")

        return {
            "enabled": True,
            "model_name": model_name,
            "device": device,
            "quantization": quantization,
            "max_memory": max_memory,
            "offload_folder": offload_folder,
            "trust_remote_code": trust_remote_code
        }

    except json.JSONDecodeError as e:
        _safe_print(f"配置文件JSON解析错误: {e}")
        return {"enabled": False}
    except Exception as e:
        _safe_print(f"加载配置文件失败: {e}")
        return {"enabled": False}
