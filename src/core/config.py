"""
应用程序配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESOURCES_DIR = PROJECT_ROOT / "resources"

# 图像索引数据库路径
IMAGE_INDEX_DIR = DATA_DIR / "image_index"

# 模型配置
MODEL_CONFIG = {
    # 调色模型
    "color_grading": {
        "model_path": MODELS_DIR / "color_grading.onnx",
        "input_size": (512, 512),
        "quantization": "int8"  # 低bit量化
    },
    # 图像特征提取模型
    "feature_extractor": {
        "model_name": "clip-vit-base-patch32",
        "embedding_dim": 512
    },
    # 文本理解模型
    "text_encoder": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "max_length": 128
    },
    # 3D生成模型
    "image_to_3d": {
        "model_path": MODELS_DIR / "image_to_3d.onnx",
        "depth_model": MODELS_DIR / "depth_estimation.onnx"
    }
}

# 调色预设
COLOR_PRESETS = {
    "蓝调": {"hue_shift": -20, "saturation": 1.1, "temperature": -15},
    "暖调": {"hue_shift": 10, "saturation": 1.05, "temperature": 20},
    "复古": {"hue_shift": 5, "saturation": 0.85, "contrast": 1.1, "fade": 0.15},
    "电影感": {"contrast": 1.2, "saturation": 0.9, "shadows": -10, "highlights": -5},
    "日系": {"exposure": 0.1, "contrast": 0.9, "saturation": 0.85, "temperature": 5},
    "黑金": {"saturation": 0.7, "split_tone_shadows": [30, 20, 10], "split_tone_highlights": [255, 215, 0]},
}


# 应用程序版本
APP_VERSION = "1.0.0"

# UI配置
UI_CONFIG = {
    "window_title": f"AI全模态影像处理 v{APP_VERSION}",
    "default_size": (1400, 900),
    "min_size": (1024, 768),
    "theme": "dark"
}

# 支持的图像格式
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".raw", ".cr2", ".nef"]

# 3D导出格式
EXPORT_3D_FORMATS = [".obj", ".gltf", ".glb", ".fbx", ".stl"]

