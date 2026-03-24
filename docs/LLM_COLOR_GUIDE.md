# 本地大模型智能调色使用指南

## 概述

本软件使用轻量化的本地大模型进行智能调色分析。与传统关键词匹配相比，大模型能够：

- **理解任何描述**：不限于预定义关键词，能理解"太阳色"、"大海"、"森林"等任何自然语言描述
- **智能判断意图**：自动识别是否为调色指令，避免将"鱼香肉丝"等无关内容误判
- **精准色彩分析**：根据事物的色彩特征生成最合适的调色参数
- **完全本地运行**：无需联网，无需API密钥，保护隐私

## 推荐模型

### Qwen2.5-1.5B-Instruct（推荐）

- **参数量**：1.5B（15亿参数）
- **模型大小**：约3GB
- **显存需求**：3-4GB（GPU）或 4-6GB（CPU）
- **特点**：轻量化，中文理解能力强，阿里巴巴开源
- **适合**：大多数用户，性能和效果的平衡点

### 其他可选模型

| 模型 | 参数量 | 大小 | 显存需求 | 说明 |
|------|--------|------|----------|------|
| Qwen2.5-3B-Instruct | 3B | ~6GB | ~6GB | 效果更好，需要更多资源 |
| Qwen2.5-7B-Instruct | 7B | ~14GB | ~14GB | 专业级效果，高配置推荐 |

## 快速开始

### 方法一：自动下载（推荐）

#### 1. 安装依赖

```bash
pip install transformers torch huggingface_hub
```

#### 2. 运行下载脚本

```bash
python scripts/download_llm.py
```

脚本会自动：
- 显示可用模型列表
- 下载选中的模型
- 提示配置方法

#### 3. 创建配置文件

复制配置文件模板：
```bash
cp llm_config.example.json llm_config.json
```

如果模型下载到了自定义位置，修改配置文件：
```json
{
  "enabled": true,
  "model_name": "path/to/your/model",
  "device": "auto"
}
```

#### 4. 启动软件

```bash
python main.py
```

首次启动会加载模型，需要等待几秒钟。

### 方法二：手动下载

#### 1. 安装依赖

```bash
pip install transformers torch
```

#### 2. 使用Python下载模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载模型
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 保存到本地
save_path = "./models/qwen2.5-1.5b"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
```

#### 3. 创建配置文件

创建 `llm_config.json`：
```json
{
  "enabled": true,
  "model_name": "./models/qwen2.5-1.5b",
  "device": "auto"
}
```

## 配置说明

### 配置文件字段

```json
{
  "enabled": true,                           // 是否启用LLM功能
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct", // 模型名称或本地路径
  "device": "auto",                          // 设备选择
  "quantization": {                          // 量化配置 (v1.0.0新增)
    "enabled": true,                         // 是否启用量化
    "bits": 4                                // 量化位数：4 或 8
  }
}
```

### device 参数

- `"auto"`：自动选择（推荐）
  - 有GPU时使用GPU
  - 无GPU时使用CPU
- `"cuda"`：强制使用GPU
- `"cpu"`：强制使用CPU

### quantization 参数 (省显存神器)

- **4-bit**: 显存占用极低（约1.5GB），速度最快，精度轻微损失（推荐大多数显卡）。
- **8-bit**: 显存占用中等（约3GB），精度接近原版。
- **关闭 (enabled: false)**: 使用半精度(FP16)或全精度(FP32)，显存占用最高。

### 使用GPU vs CPU

**GPU模式（推荐）：**
- 速度快（1-2秒响应）
- 需要NVIDIA显卡，支持CUDA
- 显存需求：
  - 4-bit量化: ~1.5GB
  - 8-bit量化: ~3GB
  - 无量化: ~4GB+

**CPU模式：**
- 速度较慢（5-10秒响应）
- 无需显卡
- 内存需求：6-8GB起

## 使用示例

配置完成后，在调色面板输入任何描述：

| 输入 | 分析结果 |
|------|----------|
| **太阳色** | 金黄色、橙色暖色调，色温+45，饱和度+25% |
| **大海** | 蓝色冷色调，色温-20，饱和度+15% |
| **森林** | 绿色调，色温偏冷，饱和度提高 |
| **日落** | 橙红暖色调，高饱和度，略提亮 |
| **冰雪** | 冷色调，提高曝光，降低色温 |
| **复古胶片** | 降低饱和度，增加颗粒感，轻微褪色 |
| **清新明亮** | 提高曝光和清晰度，略降对比度 |
| **鱼香肉丝** | ✗ 识别为非调色指令，不做处理 |

## 查看分析日志

启动软件后，控制台会显示调色分析过程：

```
正在加载本地模型: Qwen/Qwen2.5-1.5B-Instruct
✓ 检测到CUDA，使用GPU加速
✓ 模型加载成功: Qwen/Qwen2.5-1.5B-Instruct
  设备: cuda
  参数量: 1.54B

[调色指令] 处理: 太阳色
[LLM分析] 太阳呈现金黄色、橙色的暖色调，色温高，饱和度高
[LLM分析] 参数: 曝光=0.15, 对比度=1.00, 色温=45, 饱和度=1.25
```

非调色指令：
```
[调色指令] 处理: 鱼香肉丝
[LLM分析] 非调色指令: 这是一道菜名，不是色彩调整指令
```

## 性能优化

### 减少内存占用

1. 使用更小的模型（1.5B而不是3B/7B）
2. 使用CPU模式（如果显存不足）
3. 关闭其他占用GPU的程序

### 提高响应速度

1. 使用GPU加速（如果有显卡）
2. 升级到更快的GPU
3. 使用量化模型（高级用户）

## 故障排除

### 1. "需要安装transformers库"

```bash
pip install transformers torch
```

### 2. "CUDA out of memory"（显存不足）

**解决方案：**
- 使用更小的模型（1.5B）
- 切换到CPU模式：`"device": "cpu"`
- 关闭其他程序

### 3. "模型加载失败"

**可能原因：**
- 网络问题（首次下载）
- 磁盘空间不足
- 模型文件损坏

**解决方案：**
- 检查网络连接
- 确保有足够磁盘空间（至少10GB）
- 重新下载模型

### 4. "分析速度很慢"

**CPU模式下正常现象，可以：**
- 使用GPU加速
- 使用更小的模型
- 耐心等待（首次分析会慢一些，后续会快）

### 5. 回退到传统模式

如果LLM初始化失败，系统会自动回退到传统关键词匹配：

```
警告: 无法初始化本地大模型: ...
  将使用传统关键词匹配
```

## 禁用LLM功能

如果要临时禁用LLM功能，修改配置文件：

```json
{
  "enabled": false
}
```

或者删除 `llm_config.json` 文件。

## 系统要求

### 最低配置（CPU模式）
- CPU：4核心以上
- 内存：8GB以上
- 磁盘：10GB可用空间
- 响应时间：5-10秒

### 推荐配置（GPU模式）
- GPU：NVIDIA显卡，4GB显存以上
- 内存：8GB以上
- 磁盘：10GB可用空间
- 响应时间：1-2秒

### 高性能配置
- GPU：NVIDIA RTX系列，8GB显存以上
- 可使用3B或7B模型获得更好效果
- 响应时间：<1秒

## 常见问题

**Q: 需要联网吗？**
A: 首次下载模型需要联网，之后可完全离线使用。

**Q: 费用如何？**
A: 完全免费，无需API密钥，无使用限制。

**Q: 隐私安全吗？**
A: 完全本地运行，数据不会上传，保护隐私。

**Q: 可以更换其他模型吗？**
A: 可以，只需修改配置文件中的model_name为其他支持的模型。

**Q: 中文效果如何？**
A: Qwen系列模型专门优化了中文，效果很好。

**Q: 可以在Mac上使用吗？**
A: 可以，但Mac的GPU（Apple Silicon）需要额外配置，建议使用CPU模式。

## 技术支持

如遇问题，请查看：
1. 控制台输出的错误信息
2. 本文档的"故障排除"部分
3. GitHub Issues: https://github.com/your-repo/issues

