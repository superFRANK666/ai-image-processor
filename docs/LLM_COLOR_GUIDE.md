# 大模型智能调色使用指南

## 概述

一句话调色使用大模型理解自然语言，并输出结构化调色参数。当前支持：

- 本地 Transformers/HuggingFace 模型：`provider: "local"`
- OpenAI Chat Completions 格式：`provider: "openai"`
- Anthropic Messages API 格式：`provider: "anthropic"`
- OpenAI 兼容网关：`provider: "openai-compatible"`

如果大模型不可用，系统会自动回退到传统关键词和语义匹配。

配置入口：

- 推荐：在应用内打开“模型配置”，通过顶部“一句话调色后端”选择并保存。
- 高级：直接编辑 `llm_config.json`，适合批量部署或设置额外请求字段。

## 能力

- 理解开放式描述，例如“太阳色”“森林深绿色”“青橙电影感”。
- 判断非调色指令，避免把“鱼香肉丝”这类文本强行映射到调色。
- 输出丰富参数，包括基础影调、HSL、三路色轮、曲线、质感、柔光、颗粒等。
- 支持本地离线和 API 两种部署方式。

## API 后端配置

OpenAI：

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

Anthropic：

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

OpenAI 兼容网关：

```json
{
  "enabled": true,
  "provider": "openai-compatible",
  "model": "",
  "base_url": "",
  "api_key_required": false
}
```

推荐把密钥放在环境变量中，不要把真实密钥提交到仓库。

## 本地模型配置

下载推荐模型：

```bash
python scripts/download_all_models.py
```

轻量本地配置：

```json
{
  "enabled": true,
  "provider": "local",
  "model_name": "./models/Qwen2.5-1.5B-Instruct",
  "device": "auto",
  "trust_remote_code": false
}
```

7B 4-bit 量化：

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

旧配置未写 `provider` 时会按本地模型处理。

## 使用示例

| 输入 | 预期参数方向 |
|------|--------------|
| `太阳色` | 提升暖色、高光金黄、适度提高饱和 |
| `大海` | 蓝/青 HSL 增强、略冷、提高通透 |
| `森林深绿色` | 绿色增强、压黄、暗部更厚重 |
| `青橙电影感，暗部冷一点，高光像夕阳` | 阴影青蓝、高光橙金、曲线和暗角 |
| `肤色别太红，天空更蓝` | 红色饱和降低，蓝色/青色定向增强 |
| `鱼香肉丝` | 非调色指令或传统解析无明显参数 |

## 日志示例

本地模式：

```text
[OK] 加载LLM配置: provider=local, model=./models/Qwen2.5-1.5B-Instruct, device=auto
✓ 本地大模型: ./models/Qwen2.5-1.5B-Instruct 已启用（异步模式）
```

API 模式：

```text
[OK] 加载API LLM配置: provider=openai, model=your-model, base_url=https://api.openai.com/v1
✓ API大模型: openai/your-model 已启用（异步模式）
```

回退：

```text
[LLM分析] 失败: API请求失败...，回退到传统匹配
```

## 性能与隐私

本地模式：

- 可离线运行，图像和文本不发送到外部服务。
- 需要本地磁盘、内存或显存。
- 首次下载和加载耗时较长。

API 模式：

- 不需要下载本地 LLM。
- 响应速度取决于网络和服务商。
- 文本提示会发送到配置的 API 服务；请按服务商条款处理隐私和合规。

## 故障排除

API Key 缺失：

- 检查服务商默认环境变量是否已设置，例如 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`。
- 本地 OpenAI 兼容网关无需密钥时设置 `"api_key_required": false`。

API 请求失败：

- 检查 `base_url` 是否正确。
- OpenAI 兼容服务应提供 `/v1/chat/completions`。
- Anthropic 服务应提供 `/v1/messages`。
- 增大 `timeout` 或查看网关日志。

本地模型加载失败：

- 确认 `model_name` 指向完整模型目录。
- 显存不足时使用更小模型、4-bit 量化或 CPU。
- 不信任来源时保持 `trust_remote_code: false`。

禁用 LLM：

```json
{
  "enabled": false
}
```

更多配置细节见 [LLM_CONFIG_GUIDE.md](LLM_CONFIG_GUIDE.md)。
