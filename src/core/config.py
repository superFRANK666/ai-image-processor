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
    "蓝调": {
        "temperature": -16, "blue_saturation": 18, "aqua_saturation": 10,
        "shadow_hue": 218, "shadow_saturation": 14,
    },
    "暖调": {
        "temperature": 22, "orange_saturation": 10, "yellow_saturation": 8,
        "highlight_hue": 42, "highlight_saturation": 16,
    },
    "胶片复古": {
        "contrast": 1.08, "saturation": 0.84, "temperature": 8,
        "fade": 0.16, "grain": 18, "curve_shadows": 12,
        "curve_highlights": -6,
    },
    "青橙电影": {
        "contrast": 1.22, "saturation": 0.9, "curve_darks": -10,
        "curve_lights": 8, "shadow_hue": 195, "shadow_saturation": 26,
        "highlight_hue": 38, "highlight_saturation": 22, "vignette": 12,
    },
    "空气日系": {
        "exposure": 0.16, "contrast": 0.88, "saturation": 0.82,
        "shadows": 14, "curve_shadows": 10, "texture": -8,
        "blue_luminance": 8,
    },
    "奶油人像": {
        "exposure": 0.12, "contrast": 0.9, "temperature": 6,
        "orange_luminance": 12, "red_saturation": -10,
        "texture": -22, "clarity": -8, "noise_reduction": 16,
    },
    "黑金": {
        "saturation": 0.7, "contrast": 1.28, "curve_darks": -14,
        "shadow_hue": 215, "shadow_saturation": 12,
        "highlight_hue": 44, "highlight_saturation": 32,
    },
    "霓虹赛博": {
        "contrast": 1.3, "saturation": 1.2, "vibrance": 35,
        "bloom": 22, "shadow_hue": 245, "shadow_saturation": 24,
        "highlight_hue": 305, "highlight_saturation": 28,
        "blue_saturation": 18, "magenta_saturation": 24,
    },
    "蓝天通透": {
        "blue_saturation": 28, "blue_luminance": -6,
        "aqua_saturation": 14, "dehaze": 18,
        "clarity": 10, "whites": 8,
    },
    "森林绿调": {
        "temperature": -6, "tint": -8, "green_hue": -6,
        "green_saturation": 22, "green_luminance": -4,
        "yellow_saturation": -8, "midtone_detail": 10,
    },
    "低调暗黑": {
        "exposure": -0.28, "gamma": 0.82, "contrast": 1.26,
        "blacks": -22, "curve_darks": -14, "vignette": 24,
    },
    "黑白银盐": {
        "saturation": 0.0, "contrast": 1.26, "grain": 26,
        "curve_shadows": 10, "curve_darks": -8, "curve_lights": 8,
    },
    "黄金时刻": {
        "temperature": 24, "orange_saturation": 16,
        "yellow_luminance": 8, "highlight_hue": 40,
        "highlight_saturation": 26, "bloom": 8,
    },
}


# 应用程序版本
APP_VERSION = "1.2.0"

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

