"""
图像IO工具 - 支持中文路径
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def imread(filepath: str) -> Optional[np.ndarray]:
    """
    读取图像,支持中文路径

    Args:
        filepath: 图像文件路径

    Returns:
        图像numpy数组,失败返回None
    """
    try:
        filepath = str(Path(filepath).resolve())
        img_stream = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)

        if img is None:
            # 降级到常规读取
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print(f"读取图像失败 {filepath}: {e}")
        return None


def imwrite(filepath: str, image: np.ndarray) -> bool:
    """
    保存图像,支持中文路径

    Args:
        filepath: 保存路径
        image: 图像数组

    Returns:
        是否成功
    """
    try:
        filepath = str(Path(filepath).resolve())
        ext = Path(filepath).suffix.lower()

        if not ext:
            ext = '.jpg'
            filepath = filepath + ext

        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        _, img_encoded = cv2.imencode(ext, image)
        img_encoded.tofile(filepath)
        return True
    except Exception as e:
        print(f"保存图像失败 {filepath}: {e}")
        return False


def safe_imread(filepath: str, default_color=(128, 128, 128)) -> np.ndarray:
    """
    安全读取图像,失败时返回占位图

    Args:
        filepath: 文件路径
        default_color: 占位图颜色(BGR)

    Returns:
        图像数组或占位图
    """
    img = imread(filepath)
    if img is None:
        # 返回400x300灰色占位图
        img = np.full((300, 400, 3), default_color, dtype=np.uint8)

        # 添加文字提示
        text = "Failed to load image"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = (400 - text_size[0]) // 2
        text_y = (300 + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2)

    return img
