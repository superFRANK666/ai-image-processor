# 项目计划书：AI全模态影像处理软件

## 1. 项目概述

### 背景 (Pain points)
在当前的影像处理领域，用户面临以下核心痛点：
*   **学习曲线陡峭**：传统专业软件如 Photoshop 或 Lightroom 拥有成百上千个参数滑块，普通用户难以快速上手。
*   **工作流割裂**：2D 图像编辑、3D 空间重建与本地海量素材管理通常需要切换多个软件，效率低下。
*   **检索效率低**：传统的文件夹式目录管理难以通过“内容语义”快速找到特定风格或内容的旧照片。
*   **隐私风险**：依赖云端 AI 的工具存在数据泄露风险，且受限于带宽 and 网络稳定性。

### 核心能力 (Current Capabilities)
本项目已打造出一款**基于端侧 AI 技术**的现代化影像处理软件，集成以下核心能力：
*   **语义化交互**：集成 **Qwen2.5-1.5B** 本地大模型，通过自然语言直接进行专业级调色（如“调出电影感”、“日系小清新”），支持 4-bit/8-bit 量化运行。
*   **多维感知 3D化**：基于 **Depth Anything** 与 **MobileSAM** 技术，实现单张 2D 照片到高精度 3D 网格的瞬间转换，支持 360° 交互式查看与深度图可视化。
*   **智能管理**：构建本地海量影像的索引库，支持异步缩略图加载与高性能列表滚动，提供基于语义的图像检索。
*   **极致性能**：利用延迟加载、图像哈希缓存与智能缩放技术，在普通 PC 上提供流畅的实时响应体验。

### 核心亮点 (Core Highlights)
*   **Local LLM Engine**: 完整的本地大模型运行环境，支持指令解析与参数映射，无需联网即可理解复杂语义，具备显存智能管理与 CPU 卸载功能。
*   **Smart 3D Generation**: 结合深度估计与语义分割，自动剔除背景并构建 3D 实体，内置 Mesh 生成算法优化（向量化加速），生成速度提升 100 倍。
*   **Robust Architecture**: 模块化设计，具备完善的异常处理与降级机制（如缺少模型时自动回退到传统算法），确保软件稳定性。

## 2. 技术落地 (Technical Implementation)

### 总体架构
系统采用模块化的端侧架构，核心分为 UI 交互层、AI 引擎控制层与数据持久层：

1.  **UI 交互层**：
    *   基于 **PySide6** 构建的高性能界面。
    *   **Library Manager**：支持分页浏览、异步缩略图加载 (QThread)、多选操作与右键快捷菜单。
    *   **Interactive 3D**: 集成 Open3D/Visualizer 视窗（或自定义 OpenGL 组件）进行模型展示。

2.  **AI 引擎层 (Production Ready)**：
    *   **NLP Controller**: `LocalLLMColorAnalyzer` 类封装了 Qwen 模型，支持 System Prompt 定义与 JSON 格式化输出，确保指令执行的准确性。
    *   **Vision Engine**:
        *   **Depth Estimator**: 集成 `Depth Anything` (Transformer) 与 ONNX Runtime 加速。
        *   **Segmenter**: 集成 `MobileSAM`，支持点选、框选与路径分割，内置图像 Hash 缓存避免重复推理。
    *   **Performance**: 显存自动管理，支持量化配置 (BitsAndBytes) 与设备自动映射 (CUDA/CPU)。

3.  **持久层与工具**：
    *   **Image IO**: 针对中文路径优化的 `imread_safe` / `imwrite_safe`。
    *   **Storage**: 结合 ChromaDB (向量索引) 与本地文件系统管理。

### 技术栈 (Current Tech Stack)
```yaml
AI Core:
  - LLM: Qwen2.5-1.5B-Instruct (Transformers, BitsAndBytes)
  - Vision: Depth Anything (Depth), MobileSAM (Segmentation)
  - Inference: PyTorch, ONNX Runtime
  
System Services:
  - GUI: PySide6 (Qt for Python)
  - Image Processing: OpenCV (cv2), NumPy
  - Concurrency: QThread (Async UI), Multiprocessing
  
Optimization:
  - Memory: 4-bit/8-bit Quantization, CPU Offloading
  - Speed: Vectorized Numpy operations for Mesh generation
  - Caching: Image content hashing (MD5) for redundant processing avoidance
```

## 3. 演进历程与路线图

### 状态更新 (Status Update)
*   **[2025.12] v1.0.1 核心发布**: 完成 NLP 调色、3D 生成、本地库管理三大核心模块。
*   **[2026.01] v1.1.0 性能优化**: 
    *   引入 LLM 量化支持，大幅降低显存占用（<4GB 显存可用）。
    *   重构 3D 网格生成算法，从 Python 循环优化为 NumPy 向量化操作。
    *   完善依赖检查机制与降级策略（无模型时可用）。

### 未来规划 (Future Roadmap)
1.  **多视角融合 (True Multiview)**: 进一步探索从多张不同角度照片重建真实 3D 场景（NeRF/Gaussian Splatting 方向）。
2.  **端侧 LoRA 训练**: 允许用户使用本地素材微调风格模型。
3.  **视频流处理**: 将目前的单图处理能力扩展至视频关键帧，实现智能视频剪辑。

---
*文档版本：v1.1.0 (Live Updated)*
*最后更新：2026-01-07*
