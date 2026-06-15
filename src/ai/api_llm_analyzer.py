"""
API-backed large model color analyzer.

Supports OpenAI-compatible chat-completions endpoints and Anthropic Messages API
without requiring provider-specific SDKs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


DEFAULT_SYSTEM_PROMPT = """你是一个专业的色彩调色专家。请把用户的一句话描述转换为调色参数。
只返回 JSON，不要包含其它文字。JSON 必须包含字段：
is_color_related: bool
reasoning: string
parameters: object
可用参数包括 exposure, brightness, contrast, gamma, highlights, shadows, whites, blacks,
temperature, tint, saturation, vibrance, hue_shift, HSL 参数、三路色轮、clarity,
texture, sharpen, noise_reduction, dehaze, bloom, vignette, grain, fade, tone_curve 等。
如果描述不是调色指令，请返回 {"is_color_related": false, "reasoning": "...", "parameters": {}}。"""


PROVIDER_ALIASES = {
    "openai": "openai",
    "openai-compatible": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "compatible": "openai-compatible",
    "chat-completions": "openai-compatible",
    "chat_completions": "openai-compatible",
    "anthropic": "anthropic",
    "claude": "anthropic",
}

PROVIDER_DEFAULTS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
    },
    "openai-compatible": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "endpoint": "/chat/completions",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "endpoint": "/v1/messages",
        "api_version": "2023-06-01",
    },
}


def normalize_provider(provider: str) -> str:
    """Normalize provider aliases used by llm_config.json."""
    normalized = (provider or "local").strip().casefold().replace("_", "-")
    return PROVIDER_ALIASES.get(normalized, normalized)


class APILLMColorAnalyzer:
    """Use a remote API model to generate color-grading parameters."""

    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
        temperature: Optional[float] = 0.65,
        max_tokens: Optional[int] = 512,
        headers: Optional[Dict[str, str]] = None,
        api_version: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        api_key_required: bool = True,
        session=None,
    ):
        self.provider = normalize_provider(provider)
        if self.provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"不支持的 API provider: {provider}")
        if not model:
            raise ValueError("API LLM 配置缺少 model")

        defaults = PROVIDER_DEFAULTS[self.provider]
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env or defaults.get("api_key_env")
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")
        self.endpoint = endpoint or defaults["endpoint"]
        self.timeout = float(timeout or 30.0)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = dict(headers or {})
        self.api_version = api_version or defaults.get("api_version")
        self.response_format = response_format
        self.extra_body = dict(extra_body or {})
        self.api_key_required = bool(api_key_required)

        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def analyze(self, description: str) -> Dict[str, Any]:
        """Analyze a natural-language color command and return normalized JSON."""
        try:
            user_prompt = (
                "请分析以下描述并生成调色参数：\n\n"
                f'"{description}"\n\n'
                "请返回JSON格式的分析结果。"
            )
            if self.provider in ("openai", "openai-compatible"):
                response = self._call_openai_chat(user_prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic_messages(user_prompt)
            else:
                raise ValueError(f"不支持的 API provider: {self.provider}")

            result_text = self._extract_json(response)
            result = json.loads(result_text)
            if not isinstance(result, dict):
                raise ValueError("API 返回结果不是字典格式")

            result.setdefault("is_color_related", True)
            result.setdefault("reasoning", "API模型分析")
            result.setdefault("parameters", {})
            if not isinstance(result["parameters"], dict):
                result["parameters"] = {}
            return result
        except Exception as exc:
            print(f"API模型分析失败: {exc}")
            return {
                "is_color_related": False,
                "reasoning": f"API分析失败: {exc}",
                "parameters": {},
            }

    def _call_openai_chat(self, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.response_format:
            payload["response_format"] = self.response_format
        payload.update(self.extra_body)

        data = self._post_json(self._url(), payload, self._openai_headers())
        return self._extract_openai_text(data)

    def _call_anthropic_messages(self, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "system": self.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.max_tokens or 512,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.extra_body)

        data = self._post_json(self._url(), payload, self._anthropic_headers())
        return self._extract_anthropic_text(data)

    def _url(self) -> str:
        if self.endpoint.startswith("http://") or self.endpoint.startswith("https://"):
            return self.endpoint
        return f"{self.base_url}{self.endpoint if self.endpoint.startswith('/') else '/' + self.endpoint}"

    def _api_key_value(self) -> Optional[str]:
        if self.api_key:
            if self.api_key.startswith("env:"):
                return os.environ.get(self.api_key[4:])
            if self.api_key.startswith("$"):
                return os.environ.get(self.api_key[1:])
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def _openai_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.headers)
        self._ensure_auth(headers, "Authorization")
        return headers

    def _anthropic_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.api_version or "2023-06-01",
        }
        api_key = self._api_key_value()
        if api_key:
            headers["x-api-key"] = api_key
        headers.update(self.headers)
        self._ensure_auth(headers, "x-api-key")
        return headers

    def _ensure_auth(self, headers: Dict[str, str], header_name: str):
        if not self.api_key_required:
            return
        header_names = {name.casefold() for name in headers}
        if header_name.casefold() not in header_names:
            env_hint = f" 环境变量 {self.api_key_env}" if self.api_key_env else ""
            raise RuntimeError(f"未配置 API Key，请设置{env_hint} 或在配置中提供 api_key")

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            body = getattr(locals().get("response", None), "text", "")
            suffix = f"; 响应: {body[:300]}" if body else ""
            raise RuntimeError(f"API请求失败: {exc}{suffix}") from exc

        try:
            return response.json()
        except Exception as exc:
            body = getattr(response, "text", "")
            raise RuntimeError(f"API响应不是JSON: {body[:300]}") from exc

    def _extract_openai_text(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message", {}) if isinstance(first, dict) else {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") in (None, "text", "output_text")
                ]
                if text_parts:
                    return "".join(text_parts)
            text = first.get("text") if isinstance(first, dict) else None
            if isinstance(text, str):
                return text

        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text

        output = data.get("output")
        if isinstance(output, list):
            text_parts = []
            for item in output:
                for block in item.get("content", []) if isinstance(item, dict) else []:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
            if text_parts:
                return "".join(text_parts)

        raise ValueError("无法从 OpenAI 格式响应中提取文本")

    def _extract_anthropic_text(self, data: Dict[str, Any]) -> str:
        content = data.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") in (None, "text")
            ]
            if text_parts:
                return "".join(text_parts)
        raise ValueError("无法从 Anthropic 格式响应中提取文本")

    def _extract_json(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return text[start:end + 1]
        return text
