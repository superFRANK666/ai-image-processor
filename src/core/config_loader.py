"""
配置加载工具
"""
import json
import os
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


def _normalize_provider(provider: str) -> str:
    """归一化 LLM 后端名称，兼容旧配置和常见写法。"""
    normalized = (provider or "local").strip().casefold().replace("_", "-")
    aliases = {
        "huggingface": "local",
        "transformers": "local",
        "openai-compatible": "openai-compatible",
        "compatible": "openai-compatible",
        "chat-completions": "openai-compatible",
        "anthropic": "anthropic",
        "claude": "anthropic",
    }
    return aliases.get(normalized, normalized)


def _as_number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_number(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return _as_number(value, default if default is not None else 0.0)


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载大模型配置，支持本地 transformers 模型和远程 API 模型。

    Args:
        config_path: 配置文件路径，默认为项目根目录的llm_config.json

    Returns:
        配置字典。enabled=false 时使用传统关键词匹配。
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

        provider = _normalize_provider(config.get("provider", "local"))
        if provider == "local":
            return _load_local_llm_config(config)
        if provider in {"openai", "openai-compatible", "anthropic"}:
            return _load_api_llm_config(config, provider)

        _safe_print(f"不支持的 LLM provider: {provider}")
        return {"enabled": False}

    except json.JSONDecodeError as e:
        _safe_print(f"配置文件JSON解析错误: {e}")
        return {"enabled": False}
    except Exception as e:
        _safe_print(f"加载配置文件失败: {e}")
        return {"enabled": False}


def _load_local_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """读取本地 transformers 模型配置。"""
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

    info_parts = [f"provider=local", f"model={model_name}", f"device={device}"]
    if quantization.get("enabled"):
        info_parts.append(f"quant={quantization.get('bits', 4)}-bit")
    if max_memory:
        info_parts.append(f"max_mem={max_memory}")
    if trust_remote_code:
        info_parts.append("trust_remote_code=true")

    _safe_print(f"[OK] 加载LLM配置: {', '.join(info_parts)}")

    return {
        "enabled": True,
        "provider": "local",
        "model_name": model_name,
        "device": device,
        "quantization": quantization,
        "max_memory": max_memory,
        "offload_folder": offload_folder,
        "trust_remote_code": trust_remote_code,
    }


def _load_api_llm_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """读取 OpenAI/Anthropic 等 API 模型配置。"""
    model = config.get("model") or config.get("model_name")
    if not model:
        _safe_print("API LLM 配置缺少 model")
        return {"enabled": False}

    default_env = {
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]
    default_base_url = {
        "openai": "https://api.openai.com/v1",
        "openai-compatible": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
    }[provider]

    api_key = config.get("api_key")
    api_key_env = config.get("api_key_env") or default_env
    api_key_required = bool(config.get("api_key_required", True))
    headers = config.get("headers", {})
    if not isinstance(headers, dict):
        _safe_print("headers 配置格式错误，已忽略")
        headers = {}

    response_format = config.get("response_format")
    if response_format is not None and not isinstance(response_format, dict):
        _safe_print("response_format 配置格式错误，已忽略")
        response_format = None

    extra_body = config.get("extra_body", {})
    if not isinstance(extra_body, dict):
        _safe_print("extra_body 配置格式错误，已忽略")
        extra_body = {}

    has_env_key = bool(api_key_env and os.environ.get(api_key_env))
    has_header_key = any(str(name).casefold() in {"authorization", "x-api-key"} for name in headers)
    if api_key_required and not api_key and not has_env_key and not has_header_key:
        _safe_print(f"提示: 尚未检测到 {api_key_env}，启动分析时将回退到传统匹配")

    info_parts = [
        f"provider={provider}",
        f"model={model}",
        f"base_url={config.get('base_url', default_base_url)}",
    ]
    if api_key_env:
        info_parts.append(f"api_key_env={api_key_env}")

    _safe_print(f"[OK] 加载API LLM配置: {', '.join(info_parts)}")

    return {
        "enabled": True,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "api_key_env": api_key_env,
        "api_key_required": api_key_required,
        "base_url": config.get("base_url", default_base_url),
        "endpoint": config.get("endpoint"),
        "timeout": _as_number(config.get("timeout"), 30.0),
        "temperature": _as_optional_number(config.get("temperature", 0.65), 0.65),
        "max_tokens": _as_optional_int(config.get("max_tokens"), 512),
        "headers": headers,
        "api_version": config.get("api_version"),
        "response_format": response_format,
        "extra_body": extra_body,
    }
