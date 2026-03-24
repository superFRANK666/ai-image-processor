"""
图像风格分析器
用于分析图像的风格、结构、色彩和内容
"""
import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .image_retrieval import ImageFeatureExtractor

@dataclass
class StyleSummary:
    """图像风格总结"""
    structure: Dict[str, Any]  # 画面结构 (尺寸, 比例, 构图猜测)
    color: Dict[str, Any]      # 调色 (主色调, 饱和度, 亮度, 色温倾向)
    content_tags: List[str]    # 内容标签 (基于CLIP识别)
    style_tags: List[str]      # 风格标签 (基于CLIP识别)
    background: Dict[str, Any] # 背景特征 (模糊度, 复杂度)
    
    def to_string(self) -> str:
        """转换为文本描述"""
        tags = ", ".join(self.style_tags[:3])
        colors = ", ".join([f"RGB{c}" for c in self.color['dominant_colors'][:3]])
        return f"风格: {tags}\n色彩: {colors}\n结构: {self.structure['orientation']}"

class StyleAnalyzer:
    """
    图像风格分析器
    利用计算机视觉算法和CLIP模型分析图像风格
    """
    
    # 预定义的风格和内容标签用于Zero-shot分类
    STYLE_PROMPTS = [
        "minimalist", "vintage", "cyberpunk", "modern", "abstract",
        "oil painting", "sketch", "watercolor", "anime", "photorealistic",
        "dark moody", "bright airy", "high contrast", "pastel", "noir",
        "cinematic", "hdr", "flat design", "3d render", "pixel art"
    ]
    
    CONTENT_PROMPTS = [
        "landscape", "portrait", "architecture", "nature", "cityscape",
        "food", "animal", "technology", "interior", "flower",
        "street", "night", "sky", "water", "mountain", "people"
    ]

    def __init__(self, feature_extractor: Optional[ImageFeatureExtractor] = None):
        if feature_extractor:
            self.extractor = feature_extractor
        else:
            self.extractor = ImageFeatureExtractor()
            self.extractor._init_model()
            
        # 缓存文本Embedding
        self.style_embeddings = None
        self.content_embeddings = None
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        """预计算标签的Embedding"""
        if self.extractor.model is None:
            return
            
        try:
            self.style_embeddings = self.extractor.model.encode(self.STYLE_PROMPTS)
            self.content_embeddings = self.extractor.model.encode(self.CONTENT_PROMPTS)
        except Exception as e:
            print(f"预计算Embedding失败: {e}")

    def analyze(self, image: np.ndarray) -> StyleSummary:
        """
        全方位分析图像风格
        """
        # 1. 基础视觉分析 (结构/色彩/背景)
        structure = self._analyze_structure(image)
        color = self._analyze_color(image)
        background = self._analyze_background(image)
        
        # 2. 语义分析 (内容/风格) - 使用CLIP
        content_tags, style_tags = self._analyze_semantics_clip(image)
        
        # 如果CLIP不可用，使用回退逻辑
        if not style_tags:
            style_tags = self._fallback_style_analysis(image, color)
            
        return StyleSummary(
            structure=structure,
            color=color,
            content_tags=content_tags,
            style_tags=style_tags,
            background=background
        )

    def _analyze_structure(self, image: np.ndarray) -> Dict[str, Any]:
        """分析画面结构"""
        h, w = image.shape[:2]
        ratio = w / h
        
        orientation = "Square"
        if ratio > 1.2: orientation = "Landscape"
        elif ratio < 0.8: orientation = "Portrait"
        
        # 简单的三分法构图检测 (通过边缘分布猜测)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 计算重心
        M = cv2.moments(edges)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center_mass = (cx / w, cy / h)
        else:
            center_mass = (0.5, 0.5)
            
        composition = "Unknown"
        if 0.33 < center_mass[0] < 0.66:
            composition = "Centered"
        else:
            composition = "Rule of Thirds"
            
        return {
            "width": w,
            "height": h,
            "aspect_ratio": ratio,
            "orientation": orientation,
            "composition_guess": composition
        }

    def _analyze_color(self, image: np.ndarray) -> Dict[str, Any]:
        """分析色彩风格"""
        # 使用Extractor已有的方法
        dominant = self.extractor._extract_dominant_colors(image, k=3)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = np.mean(hsv[:, :, 1])
        val = np.mean(hsv[:, :, 2])
        
        # 色温估算 (简单的红蓝比)
        b, g, r = cv2.split(image)
        temp_ratio = np.mean(r) / (np.mean(b) + 1e-6)
        temperature = "Neutral"
        if temp_ratio > 1.2: temperature = "Warm"
        elif temp_ratio < 0.8: temperature = "Cool"
        
        return {
            "dominant_colors": dominant,
            "saturation_level": "High" if sat > 100 else "Low" if sat < 50 else "Medium",
            "brightness_level": "Bright" if val > 150 else "Dark" if val < 80 else "Medium",
            "temperature": temperature
        }

    def _analyze_background(self, image: np.ndarray) -> Dict[str, Any]:
        """分析背景 (通过拉普拉斯方差估算模糊度)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 简单的模糊检测
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        complexity = "Simple"
        if laplacian_var > 500: complexity = "Complex"
        elif laplacian_var > 100: complexity = "Medium"
        
        return {
            "blur_score": laplacian_var,
            "complexity": complexity
        }

    def _analyze_semantics_clip(self, image: np.ndarray) -> tuple[List[str], List[str]]:
        """使用CLIP进行Zero-shot分类"""
        if self.extractor.model is None or self.style_embeddings is None:
            return [], []
            
        try:
            # 编码图像
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            img_emb = self.extractor.model.encode(pil_img)
            
            # 计算相似度
            style_sim = np.dot(self.style_embeddings, img_emb)
            content_sim = np.dot(self.content_embeddings, img_emb)
            
            # 获取Top-K
            top_style_idx = np.argsort(-style_sim)[:3]
            top_content_idx = np.argsort(-content_sim)[:3]
            
            styles = [self.STYLE_PROMPTS[i] for i in top_style_idx]
            contents = [self.CONTENT_PROMPTS[i] for i in top_content_idx]
            
            return contents, styles
            
        except Exception as e:
            print(f"语义分析出错: {e}")
            return [], []

    def _fallback_style_analysis(self, image: np.ndarray, color_info: Dict) -> List[str]:
        """基于传统CV特征的简单风格猜测"""
        styles = []
        if color_info["saturation_level"] == "Low" and color_info["brightness_level"] == "High":
            styles.append("minimalist")
        if color_info["temperature"] == "Warm" and color_info["saturation_level"] == "Low":
            styles.append("vintage")
        if color_info["brightness_level"] == "Dark":
            styles.append("low-key")
        return styles
