# LLM 模型与 API 配置指南

应用内“模型配置”窗口现在可以直接配置一句话调色后端，并写入 `llm_config.json`。需要批量复制、部署或精细调整高级字段时，也可以手动编辑这个文件。

`llm_config.json` 支持两类后端：

- `local`: 本地 Transformers/HuggingFace 模型，适合离线、隐私优先、已有显卡或本地权重的场景。
- `openai` / `anthropic` / `openai-compatible`: 通过 API 调用大模型，适合不想下载本地 LLM、希望更快启用或接入企业网关的场景。

如果 `llm_config.json` 不存在，或配置为 `"enabled": false`，应用会自动回退到传统关键词解析。

## 快速选择

API 路线不需要下载一句话调色 LLM，只需要设置环境变量：

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

本地路线仍可使用模型配置窗口或下载脚本：

```bash
python scripts/download_all_models.py
```

在应用内配置时，进入“模型配置”：

- 顶部“一句话调色后端”选择禁用、本地模型、OpenAI、Anthropic 或 OpenAI 兼容。
- API 后端填写模型名、Base URL、API Key 等，然后点击“保存一句话调色配置”。
- 本地模型后端在下方“一句话调色意图理解模型”选择默认或自定义模型，再下载缺失模型。

## OpenAI API

```json
{
  "enabled": true,
  "provider": "openai",
  "model": "",
  "base_url": "https://api.openai.com/v1",
  "timeout": 30,
  "temperature": 0.65,
  "max_tokens": 512
}
```

## Anthropic API

```json
{
  "enabled": true,
  "provider": "anthropic",
  "model": "",
  "base_url": "https://api.anthropic.com",
  "timeout": 30,
  "temperature": 0.65,
  "max_tokens": 512
}
```

## OpenAI 兼容 API

适用于 Ollama、LM Studio、vLLM、兼容 OpenAI Chat Completions 的第三方或企业网关。

```json
{
  "enabled": true,
  "provider": "openai-compatible",
  "model": "",
  "base_url": "",
  "api_key_required": false,
  "timeout": 30,
  "temperature": 0.65,
  "max_tokens": 512
}
```

如果网关需要密钥，可填写 API Key，或手动编辑配置使用高级字段：

```json
{
  "enabled": true,
  "provider": "openai-compatible",
  "model": "your-model-name",
  "base_url": "https://your-gateway.example.com/v1",
  "api_key": "sk-..."
}
```

## 本地模型

旧配置仍然可用；未写 `provider` 时默认等同于 `"provider": "local"`。

轻量配置：

```json
{
  "enabled": true,
  "provider": "local",
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "auto",
  "trust_remote_code": false
}
```

7B 4-bit 量化配置：

```json
{
  "enabled": true,
  "provider": "local",
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "auto",
  "trust_remote_code": false,
  "quantization": {
    "enabled": true,
    "bits": 4,
    "compute_dtype": "float16"
  }
}
```

## 字段说明

| 字段 | 适用后端 | 说明 |
|------|----------|------|
| `enabled` | 全部 | 是否启用 LLM；`false` 时使用传统关键词匹配 |
| `provider` | 全部 | `local`、`openai`、`anthropic`、`openai-compatible` |
| `model_name` | local | HuggingFace 模型名或本地路径 |
| `model` | API | API 服务商或兼容网关中的模型名；窗口默认留空，需要手动填写 |
| `api_key` | API | 直接写入 Key；仅本机临时测试使用，不要提交。留空时使用服务商默认环境变量 |
| `api_key_required` | API | 本地兼容网关无需 Key 时设为 `false` |
| `base_url` | API | API 基础地址；OpenAI 兼容服务通常以 `/v1` 结尾，窗口默认留空 |
| `endpoint` | API | 可选，自定义路径或完整 URL |
| `timeout` | API | 请求超时秒数 |
| `temperature` | API | 生成随机性；设为 `null` 可不发送 |
| `max_tokens` | API | 最大输出 token 数 |
| `headers` | API | 自定义请求头 |
| `response_format` | OpenAI 格式 | 可选响应格式配置 |
| `extra_body` | API | 合并到请求 JSON 的附加字段 |
| `device` | local | `auto` / `cuda` / `cpu` |
| `quantization` | local | 4-bit/8-bit 量化配置 |
| `max_memory` | local | 显存/内存上限 |
| `offload_folder` | local | CPU 卸载目录 |
| `trust_remote_code` | local | 是否执行模型仓库自定义代码，默认建议 `false` |

## 故障排除

API 报未配置 Key：

- 确认已设置服务商默认环境变量，例如 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`。
- PowerShell 当前窗口设置的变量只对当前窗口有效，重开终端后需要重新设置。
- 本地兼容网关不需要密钥时，把 `api_key_required` 设为 `false`。

API 返回无法解析：

- 确认网关兼容 OpenAI `/v1/chat/completions` 或 Anthropic `/v1/messages`。
- 调高 `timeout`，或检查网关日志。
- 模型必须返回 JSON；应用会尽量从文本中提取第一个 JSON 对象。

本地显存不足：

- 使用更小模型，例如 7B -> 3B -> 1.5B。
- 开启 4-bit 量化。
- 改用 `device: "cpu"`，或切换到 API 后端。

恢复传统解析：

```json
{
  "enabled": false
}
```
