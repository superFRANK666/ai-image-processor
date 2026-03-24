# 本地大模型智能调色 - 快速开始

## 一键配置（3步）

### 1. 安装依赖

```bash
pip install transformers torch huggingface_hub
```

### 2. 下载模型

```bash
python scripts/download_all_models.py
```

按提示选择模型（推荐选1：Qwen2.5-1.5B，约3GB）

### 3. 创建配置

```bash
# Windows
copy llm_config.example.json llm_config.json

# Linux/Mac
cp llm_config.example.json llm_config.json
```

### 4. 启动软件

```bash
python main.py
```

## 使用示例

在调色面板输入任何描述：

- 输入："太阳色" → 金黄暖色调
- 输入："大海" → 蓝色冷色调
- 输入："森林" → 绿色自然调
- 输入："鱼香肉丝" → 自动识别为非调色指令

## 功能特点

✅ **完全本地运行** - 无需联网，无需API密钥
✅ **隐私保护** - 数据不上传，完全离线
✅ **智能理解** - 理解任何自然语言描述
✅ **自动判断** - 识别非调色指令，避免误判
✅ **轻量高效** - 1.5B参数模型，3-4GB显存即可

## 系统要求

**最低配置（CPU模式）：**
- 内存：8GB以上
- 磁盘：10GB可用空间
- 响应时间：5-10秒

**推荐配置（GPU模式）：**
- GPU：NVIDIA显卡，4GB显存
- 内存：8GB以上
- 响应时间：1-2秒

## 配置说明

`llm_config.json` 配置文件：

```json
{
  "enabled": true,                           // 启用LLM功能
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct", // 模型名称
  "device": "auto",                          // auto=自动，cuda=GPU，cpu=CPU
  "quantization": {                          // 量化配置 (推荐开启)
      "enabled": true,
      "bits": 4                              // 4-bit量化极省显存
  }
}
```

## 详细文档

查看完整文档：[docs/LLM_COLOR_GUIDE.md](docs/LLM_COLOR_GUIDE.md)

包含：
- 详细安装步骤
- 性能优化指南
- 故障排除方法
- 常见问题解答

## 故障排除

### 显存不足

```json
{
  "device": "cpu"
}
```

### 模型下载失败

1. 检查网络连接
2. 确保有足够磁盘空间
3. 使用代理或镜像站

### 禁用LLM

```json
{
  "enabled": false
}
```

或删除 `llm_config.json` 文件。

## 技术栈

- **模型**: Qwen2.5-1.5B-Instruct（阿里巴巴开源）
- **框架**: Transformers + PyTorch
- **推理**: 本地部署，支持GPU/CPU

## License

开源模型，可商用
