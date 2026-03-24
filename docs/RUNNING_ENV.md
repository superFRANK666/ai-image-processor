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
