# 大模型智能调色 - 快速开始

一句话调色可以使用 API 大模型，也可以继续使用本地模型。两种方式共用同一个 `llm_config.json`。

## 路线 A：API 快速启用

### 1. 在模型配置中选择后端

启动应用后打开“模型配置”，在顶部“一句话调色后端”选择 OpenAI、Anthropic 或 OpenAI 兼容，填写模型名、Base URL、API Key，然后点击“保存一句话调色配置”。

也可以手动创建配置：

```bash
cp llm_config.example.json llm_config.json
```

Windows 可使用：

```powershell
copy llm_config.example.json llm_config.json
```

将 `llm_config.json` 简化为一个配置对象，例如 OpenAI：

```json
{
  "enabled": true,
  "provider": "openai",
  "model": "",
  "base_url": "https://api.openai.com/v1"
}
```

或 Anthropic：

```json
{
  "enabled": true,
  "provider": "anthropic",
  "model": "",
  "base_url": "https://api.anthropic.com"
}
```

### 2. 设置 API Key

PowerShell：

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Linux/macOS：

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. 启动软件

```bash
python main.py
```

## 路线 B：本地离线模型

### 1. 在模型配置中选择本地模型

启动应用后打开“模型配置”，顶部“一句话调色后端”选择“本地模型”，下方“一句话调色意图理解模型”选择默认或自定义模型，然后点击“确认选择并下载缺失模型”。

命令行方式如下。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

```bash
python scripts/download_all_models.py
```

下载脚本会把一句话调色模型写入 `llm_config.json`，配置形如：

```json
{
  "enabled": true,
  "provider": "local",
  "model_name": "./models/Qwen2.5-1.5B-Instruct",
  "device": "auto",
  "trust_remote_code": false
}
```

### 4. 启动软件

```bash
python main.py
```

## OpenAI 兼容网关

如果使用 Ollama、LM Studio、vLLM 或企业网关：

```json
{
  "enabled": true,
  "provider": "openai-compatible",
  "model": "",
  "base_url": "",
  "api_key_required": false
}
```

## 使用示例

在调色面板输入：

- “太阳色” -> 金黄暖色调
- “大海” -> 蓝色冷色调
- “森林” -> 绿色自然调
- “青橙电影感，暗部冷一点，高光像夕阳” -> 分离色调与曲线组合
- “鱼香肉丝” -> 自动识别为非调色指令或回退传统解析

## 回退行为

以下情况会自动回退到传统关键词匹配：

- `llm_config.json` 不存在
- `"enabled": false`
- 本地模型加载失败
- API Key 缺失、请求失败或响应无法解析

详细字段说明见 [LLM_CONFIG_GUIDE.md](LLM_CONFIG_GUIDE.md)，调色能力说明见 [LLM_COLOR_GUIDE.md](LLM_COLOR_GUIDE.md)。
