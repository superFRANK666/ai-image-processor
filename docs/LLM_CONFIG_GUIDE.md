# LLM模型配置指南

## 快速开始

是的，您可以**直接修改 `llm_config.json` 文件**来更换模型！

### 步骤

1. **选择配置**: 从 `llm_config_examples.json` 中选择适合您硬件的配置
2. **复制配置**: 将选中的配置复制到 `llm_config.json` 文件
3. **重启程序**: 关闭并重新启动应用程序

## 推荐配置

### 🚀 推荐: 7B模型 (4-bit量化)
```json
{
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "device": "auto",
  "quantization": {
    "enabled": true,
    "bits": 4,
    "compute_dtype": "float16"
  }
}
```
**适用场景**: 8GB+ 显存，平衡性能与效果  
**显存占用**: 约 4-5GB

---

### 💪 进阶: 14B模型 (4-bit量化)
```json
{
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-14B-Instruct",
  "device": "auto",
  "quantization": {
    "enabled": true,
    "bits": 4,
    "compute_dtype": "float16"
  },
  "max_memory": {
    "0": "10GB",
    "cpu": "20GB"
  }
}
```
**适用场景**: 12GB+ 显存，追求最佳效果  
**显存占用**: 约 8-10GB

---

### 🔥 旗舰: 32B模型 (4-bit量化)
```json
{
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-32B-Instruct",
  "device": "auto",
  "quantization": {
    "enabled": true,
    "bits": 4,
    "compute_dtype": "bfloat16"
  },
  "max_memory": {
    "0": "20GB",
    "cpu": "40GB"
  },
  "offload_folder": "./model_offload"
}
```
**适用场景**: 24GB+ 显存，最强性能  
**显存占用**: 约 18-20GB

---

### 💾 轻量: 1.5B模型 (当前默认)
```json
{
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "auto"
}
```
**适用场景**: 低显存设备  
**显存占用**: 约 2GB

---

## 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `enabled` | 是否启用LLM | `true` / `false` |
| `model_name` | 模型名称 | Qwen系列或本地路径 |
| `device` | 运行设备 | `auto` / `cuda` / `cpu` |
| `quantization.enabled` | 是否量化 | `true` / `false` |
| `quantization.bits` | 量化位数 | `4` (更省显存) / `8` (更高精度) |
| `quantization.compute_dtype` | 计算精度 | `float16` / `bfloat16` |
| `max_memory` | 显存限制 | 如 `{"0": "10GB", "cpu": "20GB"}` |
| `offload_folder` | CPU卸载目录 | 文件夹路径 |

## 可用的Qwen模型

- `Qwen/Qwen2.5-1.5B-Instruct` - 1.5B参数 (最轻量)
- `Qwen/Qwen2.5-3B-Instruct` - 3B参数
- `Qwen/Qwen2.5-7B-Instruct` - 7B参数 (推荐)
- `Qwen/Qwen2.5-14B-Instruct` - 14B参数 (高性能)
- `Qwen/Qwen2.5-32B-Instruct` - 32B参数 (旗舰)
- `Qwen/Qwen2.5-72B-Instruct` - 72B参数 (超旗舰，需多卡或大量CPU卸载)

## 使用本地模型

如果您已经下载了模型到本地，可以直接指定路径：

```json
{
  "enabled": true,
  "model_name": "/path/to/models/Qwen2.5-7B-Instruct",
  "device": "cuda",
  "quantization": {
    "enabled": true,
    "bits": 4,
    "compute_dtype": "float16"
  }
}
```

## 显存不足？

### 方案1: 使用更小的模型
- 从 7B → 3B → 1.5B

### 方案2: 启用更激进的量化
```json
{
  "quantization": {
    "enabled": true,
    "bits": 4,  // 从8改为4
    "compute_dtype": "float16"
  }
}
```

### 方案3: 限制显存使用 + CPU卸载
```json
{
  "max_memory": {
    "0": "6GB",     // 限制GPU显存
    "cpu": "16GB"   // 允许CPU内存
  },
  "offload_folder": "./model_offload"  // 启用CPU卸载
}
```

### 方案4: 使用CPU（慢但可用）
```json
{
  "enabled": true,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "device": "cpu"
}
```

## 首次使用注意事项

1. **自动下载**: 首次使用会从 HuggingFace 自动下载模型
2. **耐心等待**: 模型较大(7B约14GB)，下载需要一些时间
3. **网络环境**: 建议配置 HuggingFace 镜像加速
4. **磁盘空间**: 确保有足够的磁盘空间存储模型

## 故障排除

### Q: 显存不足错误？
A: 使用更小的模型或启用4-bit量化

### Q: 模型下载失败？
A: 检查网络连接，或手动下载模型到本地后指定路径

### Q: 程序启动慢？
A: 正常现象，大模型加载需要时间。首次启动会更慢(下载+加载)

### Q: 想恢复默认？
A: 将配置改为：
```json
{
  "enabled": false
}
```

## 性能对比

| 模型 | 参数量 | 4-bit显存 | 推理速度 | 效果 |
|------|--------|-----------|----------|------|
| 1.5B | 1.5B | ~2GB | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ |
| 3B | 3B | ~3GB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| 7B | 7B | ~5GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| 14B | 14B | ~9GB | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ |
| 32B | 32B | ~18GB | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ |

---

**建议**: 从推荐的 **7B模型(4-bit量化)** 开始尝试，它在性能和效果之间有很好的平衡！

