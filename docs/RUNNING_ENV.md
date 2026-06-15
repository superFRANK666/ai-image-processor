# 运行环境说明

本项目的所有依赖库均已安装在项目根目录下的虚拟环境 `venv` 中。为了确保程序正常运行，请务必使用虚拟环境中的 Python 解释器。

## 目录结构
```text
AIImageProcessor/
├── venv/             # 包含所有已安装依赖的虚拟环境
├── main.py
└── ...
```

## 如何运行

请按照以下步骤在命令行中运行程序：

### 1. Windows (PowerShell)

**方法一：直接使用虚拟环境解释器（推荐）**

在项目根目录下，直接调用 `venv` 中的 python 执行脚本：

```powershell
.\venv\Scripts\python.exe main.py
```

**方法二：激活虚拟环境后运行**

1. 激活虚拟环境：
   ```powershell
   .\venv\Scripts\activate
   ```
   激活成功后，命令行提示符左侧会出现 `(venv)` 字样。

2. 运行程序：
   ```powershell
   python main.py
   ```

### 2. 注意事项

*   **不要直接使用系统全局的 `python main.py`**，除非你确定你的全局 Python 环境中已经安装了所有必要的依赖库。
*   如果遇到 `ModuleNotFoundError` 错误，请首先检查是否已经正确使用了虚拟环境。

## 可选 API 大模型环境变量

如果 `llm_config.json` 使用 API 后端，请在启动前设置对应环境变量：

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

OpenAI 兼容本地网关不需要密钥时，可以在配置中设置 `"api_key_required": false`。

## 本地生成目录

以下目录用于本机运行、模型缓存或测试输出，默认不提交到 Git：

*   `venv/`: 本地虚拟环境。
*   `models/`: 通过模型配置或下载脚本获取的模型权重。
*   `data/`: 本地图像索引和 ChromaDB 数据。
*   `artifacts/`: UI 截图、测试素材和临时输出。
*   `__pycache__/`、`.pytest_cache/`: Python 和测试缓存，可随时删除。

提交前建议运行 `git status --ignored -sb`，确认只有源码、文档、配置示例和测试变更进入版本库。
