"""
图像特征提取与检索系统
基于CLIP模型实现图像特征提取和相似度匹配

改进版本:
- 支持多语言CLIP模型 (中文、英文等)
- 图像预处理增强 (对比度、清晰度、主体检测)
- 多特征综合搜索 (语义特征 + 颜色特征 + 纹理特征)
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass, field
import cv2
import os
import logging
import sys

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe

# 配置日志
logger = logging.getLogger(__name__)


try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# 多语言CLIP模型配置
MULTILINGUAL_CLIP_MODELS = {
    # 推荐: 多语言模型，支持中文
    "multilingual": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    # 备选: 原始英文模型
    "english": "sentence-transformers/clip-ViT-B-32",
    # 大型多语言模型 (更准确但更慢)
    "multilingual-large": "sentence-transformers/clip-ViT-L-14-multilingual-v1",
}


@dataclass
class ImageFeature:
    """图像特征数据"""
    image_path: str
    embedding: np.ndarray  # 语义特征 (CLIP)
    color_histogram: np.ndarray  # 颜色直方图特征
    dominant_colors: List[Tuple[int, int, int]]  # 主色调
    metadata: Dict[str, Any]
    # 新增特征
    texture_features: np.ndarray = field(default_factory=lambda: np.array([]))  # 纹理特征
    edge_features: np.ndarray = field(default_factory=lambda: np.array([]))  # 边缘特征


class ImagePreprocessor:
    """
    图像预处理增强器
    在特征提取前对图像进行优化处理
    """

    @staticmethod
    def enhance_for_feature_extraction(image: np.ndarray) -> np.ndarray:
        """
        对图像进行预处理增强，提高特征提取质量

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            增强后的图像
        """
        if image is None or image.size == 0:
            return image

        enhanced = image.copy()

        # 1. 自动对比度增强 (CLAHE) - 这是一个相对快速且有效的操作
        enhanced = ImagePreprocessor._apply_clahe(enhanced)

        # 2. 去噪处理 (可选，因为比较慢)
        # enhanced = ImagePreprocessor._denoise(enhanced)

        # 3. 锐化处理
        enhanced = ImagePreprocessor._sharpen(enhanced)

        return enhanced

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """应用自适应直方图均衡化 (CLAHE)"""
        try:
            # 转换到 LAB 颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 对 L 通道应用 CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # 合并通道
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"CLAHE failed: {e}")
            return image

    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """轻度去噪处理"""
        try:
            # 使用双边滤波保持边缘
            return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        except Exception:
            return image

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        """轻度锐化处理"""
        try:
            # 锐化核
            kernel = np.array([
                [0, -0.5, 0],
                [-0.5, 3, -0.5],
                [0, -0.5, 0]
            ])
            return cv2.filter2D(image, -1, kernel)
        except Exception:
            return image

    @staticmethod
    def detect_main_subject(image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        检测图像主体区域

        Returns:
            (x, y, w, h) 主体边界框
        """
        try:
            h, w = image.shape[:2]

            # 使用显著性检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)

            # 形态学操作连接边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)

            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 找到最大的轮廓
                largest = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest)

                # 扩大边界框
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                cw = min(w - x, cw + 2 * margin)
                ch = min(h - y, ch + 2 * margin)

                return (x, y, cw, ch)

            return (0, 0, w, h)
        except Exception:
            h, w = image.shape[:2]
            return (0, 0, w, h)

    @staticmethod
    def crop_main_subject(image: np.ndarray, min_ratio: float = 0.3) -> np.ndarray:
        """
        裁剪图像主体区域

        Args:
            image: 输入图像
            min_ratio: 最小裁剪比例 (防止裁剪太小)

        Returns:
            裁剪后的图像
        """
        try:
            h, w = image.shape[:2]
            x, y, cw, ch = ImagePreprocessor.detect_main_subject(image)

            # 检查裁剪区域是否过小
            if cw * ch < w * h * min_ratio:
                return image

            return image[y:y+ch, x:x+cw]
        except Exception:
            return image


class ImageFeatureExtractor:
    """
    图像特征提取器
    提取视觉特征和语义特征

    改进版本:
    - 支持多语言 CLIP 模型
    - 图像预处理增强
    - 多种特征综合提取
    """

    def __init__(self, model_name: str = "multilingual", use_gpu: bool = True,
                 local_model_path: Optional[str] = None,
                 enable_preprocessing: bool = True):
        """
        初始化特征提取器

        Args:
            model_name: 使用的模型名称 ("multilingual", "english", "multilingual-large")
            use_gpu: 是否使用GPU
            local_model_path: 本地模型路径(如果有的话)
            enable_preprocessing: 是否启用图像预处理增强
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.local_model_path = local_model_path
        self.enable_preprocessing = enable_preprocessing
        self.model = None
        self.embedding_dim = 512
        self._model_init_attempted = False
        self._is_multilingual = False  # 标记是否为多语言模型

        # 检查本地模型路径
        if local_model_path is None:
            project_dir = Path(__file__).parent.parent.parent
            # 优先使用多语言模型
            multilingual_path = project_dir / "models" / "clip-ViT-B-32-multilingual-v1"
            english_path = project_dir / "models" / "clip-ViT-B-32"

            if multilingual_path.exists():
                self.local_model_path = str(multilingual_path)
                self._is_multilingual = True
            elif english_path.exists() and (english_path / "0_CLIPModel" / "model.safetensors").exists():
                self.local_model_path = str(english_path)

    def _init_model(self):
        """延迟初始化CLIP模型 (支持多语言)"""
        if self._model_init_attempted:
            return

        self._model_init_attempted = True

        if SentenceTransformer is None:
            logger.warning("无法加载文本编码器: No module named 'sentence_transformers'")
            print("警告: 无法加载文本编码器: No module named 'sentence_transformers'")
            return

        # 尝试加载模型的顺序
        models_to_try = []

        if self.local_model_path:
            models_to_try.append(("local", self.local_model_path))

        # 根据配置添加在线模型
        if self.model_name in MULTILINGUAL_CLIP_MODELS:
            models_to_try.append(("online", MULTILINGUAL_CLIP_MODELS[self.model_name]))
        else:
            # 默认尝试多语言模型
            models_to_try.append(("online", MULTILINGUAL_CLIP_MODELS["multilingual"]))
            models_to_try.append(("online", MULTILINGUAL_CLIP_MODELS["english"]))

        for source, model_path in models_to_try:
            try:
                print(f"尝试加载CLIP模型: {model_path} ({source})")

                # 对于在线模型，允许下载
                if source == "online":
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                else:
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'

                self.model = SentenceTransformer(model_path)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()

                # 检测是否为多语言模型
                if "multilingual" in model_path.lower():
                    self._is_multilingual = True

                print(f"CLIP模型加载成功: {model_path}")
                print(f"  - 多语言支持: {'是' if self._is_multilingual else '否'}")
                print(f"  - 特征维度: {self.embedding_dim}")
                return

            except Exception as e:
                logger.error(f"模型加载失败 ({model_path}): {e}")
                print(f"模型加载失败 ({model_path}): {e}")
                continue

        print("所有CLIP模型加载失败, 将使用传统特征提取")
        self.model = None

    def extract_features(self, image: np.ndarray, image_path: str = "",
                        use_enhancement: bool = True) -> ImageFeature:
        """
        提取图像特征 (增强版)

        Args:
            image: 输入图像 (BGR格式)
            image_path: 图像路径
            use_enhancement: 是否使用图像增强

        Returns:
            ImageFeature: 图像特征对象
        """
        # 图像预处理增强
        if self.enable_preprocessing and use_enhancement:
            enhanced_image = ImagePreprocessor.enhance_for_feature_extraction(image)
        else:
            enhanced_image = image

        # 提取语义特征 (CLIP embedding)
        embedding = self._extract_semantic_embedding(enhanced_image)

        # 提取颜色直方图
        color_histogram = self._extract_color_histogram(image)

        # 提取主色调
        dominant_colors = self._extract_dominant_colors(image)

        # 提取元数据
        metadata = self._extract_metadata(image)

        # 提取纹理特征 (新增)
        texture_features = self._extract_texture_features(image)

        # 提取边缘特征 (新增)
        edge_features = self._extract_edge_features(image)

        return ImageFeature(
            image_path=image_path,
            embedding=embedding,
            color_histogram=color_histogram,
            dominant_colors=dominant_colors,
            metadata=metadata,
            texture_features=texture_features,
            edge_features=edge_features
        )

    def _extract_semantic_embedding(self, image: np.ndarray) -> np.ndarray:
        """提取语义embedding (支持增强图像)"""
        # 确保模型已初始化
        if not self._model_init_attempted:
            self._init_model()

        if self.model is not None:
            try:
                # 转换为RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_image = Image.fromarray(image_rgb)

                # 使用CLIP编码图像
                embedding = self.model.encode(pil_image)
                return embedding
            except Exception as e:
                logger.error(f"语义特征提取失败: {e}")
                print(f"语义特征提取失败: {e}")

        # 回退到传统特征
        return self._extract_traditional_features(image)

    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取纹理特征 (用于综合搜索)

        使用 Gabor 滤波器和 LBP 结合
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            features = []

            # Gabor 滤波器响应
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=5,
                    theta=theta,
                    lambd=10,
                    gamma=0.5
                )
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.extend([np.mean(filtered), np.std(filtered)])

            # LBP 直方图
            lbp_hist = self._compute_lbp_histogram(gray, num_bins=64)
            features.extend(lbp_hist[:32])  # 取前32个bin

            return np.array(features, dtype=np.float32)
        except Exception:
            return np.array([])

    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取边缘特征 (用于综合搜索)

        使用 Canny 边缘检测和 HOG 特征简化版
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))

            features = []

            # Canny 边缘统计
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features.append(edge_density)

            # 边缘方向直方图 (简化版 HOG)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            magnitude = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx) * 180 / np.pi + 180

            # 计算方向直方图 (8 bins)
            hist, _ = np.histogram(angle, bins=8, range=(0, 360), weights=magnitude)
            hist = hist / (hist.sum() + 1e-6)
            features.extend(hist)

            return np.array(features, dtype=np.float32)
        except Exception:
            return np.array([])

    def _extract_traditional_features(self, image: np.ndarray) -> np.ndarray:
        """传统特征提取 (作为回退方案)"""
        features = []

        # 1. 颜色矩 (均值、标准差、偏度)
        for i in range(3):
            channel = image[:, :, i].flatten().astype(np.float32)
            features.extend([np.mean(channel), np.std(channel)])

        # 2. Hu矩 (形状特征)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend(hu_moments)

        # 3. 纹理特征 (使用LBP的简化版本)
        resized = cv2.resize(gray, (64, 64))
        lbp_hist = self._compute_lbp_histogram(resized)
        features.extend(lbp_hist)

        # 填充到目标维度
        feature_array = np.array(features, dtype=np.float32)
        if len(feature_array) < self.embedding_dim:
            feature_array = np.pad(feature_array, (0, self.embedding_dim - len(feature_array)))
        else:
            feature_array = feature_array[:self.embedding_dim]

        # 归一化
        norm = np.linalg.norm(feature_array)
        if norm > 0:
            feature_array = feature_array / norm

        return feature_array

    def _compute_lbp_histogram(self, gray: np.ndarray, num_bins: int = 256) -> np.ndarray:
        """计算LBP直方图"""
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i - 1, j - 1] >= center) << 7
                code |= (gray[i - 1, j] >= center) << 6
                code |= (gray[i - 1, j + 1] >= center) << 5
                code |= (gray[i, j + 1] >= center) << 4
                code |= (gray[i + 1, j + 1] >= center) << 3
                code |= (gray[i + 1, j] >= center) << 2
                code |= (gray[i + 1, j - 1] >= center) << 1
                code |= (gray[i, j - 1] >= center) << 0
                lbp[i - 1, j - 1] = code

        hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        return hist

    def _extract_color_histogram(self, image: np.ndarray, bins: int = 64) -> np.ndarray:
        """提取颜色直方图"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 计算HSV直方图
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

        # 归一化
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        return np.concatenate([h_hist, s_hist, v_hist])

    def _extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """提取主色调"""
        # 调整图像大小以加速
        small = cv2.resize(image, (100, 100))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # 使用K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 按出现频率排序
        counts = np.bincount(labels.flatten())
        sorted_indices = np.argsort(-counts)

        dominant_colors = []
        for idx in sorted_indices:
            color = centers[idx].astype(int)
            dominant_colors.append(tuple(color.tolist()))

        return dominant_colors

    def _extract_metadata(self, image: np.ndarray) -> Dict[str, Any]:
        """提取图像元数据"""
        h, w = image.shape[:2]

        # 亮度统计
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # 饱和度统计
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])

        # 对比度统计
        contrast = np.std(gray)

        return {
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "brightness": float(brightness),
            "saturation": float(saturation),
            "contrast": float(contrast),
        }


class ImageIndexDatabase:
    """
    图像索引数据库
    使用ChromaDB存储和检索图像特征
    """

    def __init__(self, db_path: Path):
        """
        初始化数据库

        Args:
            db_path: 数据库存储路径
        """
        self.db_path = db_path
        self.collection = None
        self.feature_extractor = ImageFeatureExtractor()

        self._init_database()

    def _init_database(self):
        """初始化ChromaDB"""
        if chromadb is None:
            print("ChromaDB未安装,将使用内存索引")
            self.memory_index = []
            return

        try:
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client.get_or_create_collection(
                name="image_features",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            print(f"ChromaDB初始化失败: {e}")
            self.memory_index = []

    def add_image(self, image_path: str, image: Optional[np.ndarray] = None, group: str = "默认") -> str:
        """
        添加图像到索引 (增强版：存储多种特征)

        Args:
            image_path: 图像路径
            image: 图像数据(可选,不提供则从路径读取)
            group: 图像分组

        Returns:
            图像ID
        """
        # 确保 CLIP 模型已加载（用于语义搜索）
        if not self.feature_extractor._model_init_attempted:
            self.feature_extractor._init_model()

        if image is None:
            image = imread_safe(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")

        # 提取特征 (包含新的纹理和边缘特征)
        features = self.feature_extractor.extract_features(image, image_path)

        # 生成唯一ID
        image_id = hashlib.md5(image_path.encode()).hexdigest()

        # 准备元数据 (包含所有特征信息)
        metadata = {
            "path": image_path,
            "group": group,
            "color_histogram": json.dumps(features.color_histogram.tolist()),
            "dominant_colors": json.dumps(features.dominant_colors),
            # 新增特征存储
            "texture_features": json.dumps(features.texture_features.tolist()) if features.texture_features.size > 0 else "[]",
            "edge_features": json.dumps(features.edge_features.tolist()) if features.edge_features.size > 0 else "[]",
            **features.metadata
        }

        # 存储到数据库
        if self.collection is not None:
            # 先尝试删除已存在的记录（避免重复）
            try:
                self.collection.delete(ids=[image_id])
            except Exception:
                pass

            self.collection.add(
                ids=[image_id],
                embeddings=[features.embedding.tolist()],
                metadatas=[metadata]
            )
        else:
            self.memory_index.append({
                "id": image_id,
                "embedding": features.embedding,
                "texture_features": features.texture_features,
                "edge_features": features.edge_features,
                "metadata": metadata
            })

        return image_id

    def rebuild_all_indexes(self, progress_callback=None) -> int:
        """
        重建所有图片的特征索引（使用CLIP模型）

        Args:
            progress_callback: 进度回调函数 (current, total)

        Returns:
            重建的图片数量
        """
        # 确保 CLIP 模型已加载
        if not self.feature_extractor._model_init_attempted:
            self.feature_extractor._init_model()

        if self.feature_extractor.model is None:
            print("警告: CLIP模型未加载，无法重建语义索引")
            return 0

        # 获取所有现有图片
        all_images = []
        if self.collection is not None:
            try:
                results = self.collection.get(include=['metadatas'])
                if results and results['ids']:
                    for i, image_id in enumerate(results['ids']):
                        if image_id.startswith("__group__"):
                            continue
                        metadata = results['metadatas'][i]
                        path = metadata.get("path", "")
                        if path and Path(path).exists():
                            all_images.append({
                                "id": image_id,
                                "path": path,
                                "group": metadata.get("group", "默认"),
                                "name": metadata.get("name", "")
                            })
            except Exception as e:
                print(f"获取图片列表失败: {e}")
                return 0

        total = len(all_images)
        print(f"[重建索引] 开始重建 {total} 张图片的特征索引...")

        rebuilt = 0
        for i, img_info in enumerate(all_images):
            try:
                image = cv2.imread(img_info["path"])
                if image is None:
                    continue

                # 重新提取特征并更新
                features = self.feature_extractor.extract_features(image, img_info["path"])

                # 更新数据库
                if self.collection is not None:
                    # 删除旧记录
                    self.collection.delete(ids=[img_info["id"]])

                    # 添加新记录 (包含所有新特征)
                    metadata = {
                        "path": img_info["path"],
                        "group": img_info["group"],
                        "name": img_info["name"],
                        "color_histogram": json.dumps(features.color_histogram.tolist()),
                        "dominant_colors": json.dumps(features.dominant_colors),
                        # 新增特征存储
                        "texture_features": json.dumps(features.texture_features.tolist()) if features.texture_features.size > 0 else "[]",
                        "edge_features": json.dumps(features.edge_features.tolist()) if features.edge_features.size > 0 else "[]",
                        **features.metadata
                    }
                    self.collection.add(
                        ids=[img_info["id"]],
                        embeddings=[features.embedding.tolist()],
                        metadatas=[metadata]
                    )

                rebuilt += 1

                if progress_callback:
                    progress_callback(i + 1, total)

                print(f"  [{i+1}/{total}] 已重建: {Path(img_info['path']).name}")

            except Exception as e:
                print(f"  [{i+1}/{total}] 失败: {img_info['path']} - {e}")

        print(f"[重建索引] 完成，成功重建 {rebuilt}/{total} 张图片")
        return rebuilt

    def search_similar(self, query_image: np.ndarray, top_k: int = 5,
                       filter_criteria: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        搜索相似图像

        Args:
            query_image: 查询图像
            top_k: 返回结果数量
            filter_criteria: 过滤条件

        Returns:
            相似图像列表
        """
        # 提取查询特征
        query_features = self.feature_extractor.extract_features(query_image)

        if self.collection is not None:
            results = self.collection.query(
                query_embeddings=[query_features.embedding.tolist()],
                n_results=top_k,
                where=filter_criteria
            )

            similar_images = []
            for i, image_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0

                similar_images.append({
                    "id": image_id,
                    "path": metadata.get("path", ""),
                    "similarity": 1 - distance,  # 转换为相似度
                    "metadata": metadata
                })

            return similar_images
        else:
            # 内存索引搜索
            return self._memory_search(query_features.embedding, top_k)

    def _memory_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """内存索引搜索"""
        if not self.memory_index:
            return []

        similarities = []
        for item in self.memory_index:
            sim = np.dot(query_embedding, item["embedding"])
            similarities.append((sim, item))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, item in similarities[:top_k]:
            results.append({
                "id": item["id"],
                "path": item["metadata"].get("path", ""),
                "similarity": float(sim),
                "metadata": item["metadata"]
            })

        return results

    def search_by_text(self, text_query: str, top_k: int = 5,
                      use_multi_feature: bool = True) -> List[Dict[str, Any]]:
        """
        通过文本描述搜索图像（增强版：多特征综合搜索）

        改进:
        - 支持多语言CLIP直接搜索中文
        - 结合语义、颜色、纹理等多特征综合评分
        - 动态调整各特征权重

        Args:
            text_query: 文本查询
            top_k: 返回数量
            use_multi_feature: 是否使用多特征综合搜索

        Returns:
            匹配的图像列表
        """
        results = []
        seen_ids = set()

        # 1. 首先进行名称/路径匹配搜索
        name_results = self.search_by_name(text_query, top_k)
        for item in name_results:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                item['match_type'] = 'name'
                results.append(item)

        # 如果名称搜索已找到足够结果，直接返回
        if len(results) >= top_k:
            return results[:top_k]

        # 2. 如果 CLIP 模型可用，进行语义搜索
        if not self.feature_extractor._model_init_attempted:
            self.feature_extractor._init_model()

        # 语义搜索相似度阈值 (多语言模型可以降低阈值)
        SIMILARITY_THRESHOLD = 0.20 if self.feature_extractor._is_multilingual else 0.23

        if self.feature_extractor.model is not None:
            try:
                # 多语言模型直接使用原文搜索，否则翻译
                if self.feature_extractor._is_multilingual:
                    search_query = text_query
                    print(f"[搜索] 使用多语言CLIP直接搜索: {text_query}")
                else:
                    search_query = self._translate_to_english(text_query)
                    print(f"[搜索] 使用CLIP语义搜索: {text_query} -> {search_query}")

                text_embedding = self.feature_extractor.model.encode(search_query)

                if self.collection is not None:
                    # 获取更多候选结果用于多特征重排序
                    n_candidates = min(top_k * 5, 100) if use_multi_feature else min(top_k * 3, 50)

                    query_results = self.collection.query(
                        query_embeddings=[text_embedding.tolist()],
                        n_results=n_candidates,
                        include=['metadatas', 'distances', 'embeddings']
                    )

                    candidates = []
                    for i, image_id in enumerate(query_results['ids'][0]):
                        if image_id in seen_ids:
                            continue
                        if image_id.startswith("__group__"):
                            continue

                        metadata = query_results['metadatas'][0][i]
                        if not metadata.get("path"):
                            continue

                        distance = query_results['distances'][0][i] if 'distances' in query_results else 0
                        semantic_similarity = 1 - distance

                        candidates.append({
                            "id": image_id,
                            "path": metadata.get("path", ""),
                            "semantic_similarity": semantic_similarity,
                            "metadata": metadata,
                            "embedding": query_results.get('embeddings', [[]])[0][i] if query_results.get('embeddings') else None
                        })

                    # 3. 多特征综合评分
                    if use_multi_feature and candidates:
                        candidates = self._multi_feature_rerank(
                            candidates, text_query, text_embedding
                        )

                    # 过滤和添加结果
                    for candidate in candidates:
                        final_score = candidate.get('final_score', candidate['semantic_similarity'])

                        print(f"  - {Path(candidate['path']).name}: "
                              f"语义={candidate['semantic_similarity']:.3f}, "
                              f"综合={final_score:.3f}")

                        if final_score < SIMILARITY_THRESHOLD:
                            continue

                        seen_ids.add(candidate['id'])
                        results.append({
                            "id": candidate['id'],
                            "path": candidate['path'],
                            "similarity": final_score,
                            "semantic_similarity": candidate['semantic_similarity'],
                            "metadata": candidate['metadata'],
                            "match_type": 'semantic'
                        })

                        if len(results) >= top_k:
                            break
                else:
                    semantic_results = self._memory_search(text_embedding, top_k * 3)
                    for item in semantic_results:
                        if item['id'] not in seen_ids:
                            if item.get('similarity', 0) < SIMILARITY_THRESHOLD:
                                continue
                            seen_ids.add(item['id'])
                            item['match_type'] = 'semantic'
                            results.append(item)
                            if len(results) >= top_k:
                                break

            except Exception as e:
                print(f"语义搜索失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[搜索] CLIP模型未加载，仅使用名称搜索")

        # 按相似度排序后返回
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        print(f"[搜索] 找到 {len(results)} 个结果")
        return results[:top_k]

    def _multi_feature_rerank(self, candidates: List[Dict], text_query: str,
                             text_embedding: np.ndarray) -> List[Dict]:
        """
        多特征综合重排序

        综合语义、颜色、纹理等多种特征进行评分

        Args:
            candidates: 候选图像列表
            text_query: 原始文本查询
            text_embedding: 文本embedding

        Returns:
            重排序后的候选列表
        """
        # 特征权重配置
        WEIGHTS = {
            'semantic': 0.6,      # 语义特征权重
            'color': 0.2,         # 颜色特征权重
            'brightness': 0.1,    # 亮度特征权重
            'texture': 0.1,       # 纹理特征权重
        }

        # 解析查询中的颜色关键词
        color_keywords = self._extract_color_keywords(text_query)
        brightness_keywords = self._extract_brightness_keywords(text_query)

        for candidate in candidates:
            scores = {
                'semantic': candidate['semantic_similarity']
            }

            metadata = candidate['metadata']

            # 颜色匹配评分
            if color_keywords:
                color_score = self._compute_color_match_score(metadata, color_keywords)
                scores['color'] = color_score
            else:
                scores['color'] = 0.5  # 中性分数

            # 亮度匹配评分
            if brightness_keywords:
                brightness = metadata.get('brightness', 128)
                brightness_score = self._compute_brightness_match_score(brightness, brightness_keywords)
                scores['brightness'] = brightness_score
            else:
                scores['brightness'] = 0.5

            # 纹理复杂度评分 (根据查询需求)
            contrast = metadata.get('contrast', 50)
            scores['texture'] = min(contrast / 100, 1.0)

            # 计算加权综合得分
            final_score = sum(WEIGHTS[k] * scores.get(k, 0.5) for k in WEIGHTS)
            candidate['final_score'] = final_score
            candidate['score_breakdown'] = scores

        # 按综合得分排序
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates

    def _extract_color_keywords(self, text: str) -> List[str]:
        """从查询文本中提取颜色关键词"""
        color_map = {
            '红': 'red', '红色': 'red',
            '蓝': 'blue', '蓝色': 'blue',
            '绿': 'green', '绿色': 'green',
            '黄': 'yellow', '黄色': 'yellow',
            '白': 'white', '白色': 'white',
            '黑': 'black', '黑色': 'black',
            '橙': 'orange', '橙色': 'orange',
            '紫': 'purple', '紫色': 'purple',
            '粉': 'pink', '粉色': 'pink',
            '灰': 'gray', '灰色': 'gray',
            '棕': 'brown', '棕色': 'brown',
            'red': 'red', 'blue': 'blue', 'green': 'green',
            'yellow': 'yellow', 'white': 'white', 'black': 'black',
            'orange': 'orange', 'purple': 'purple', 'pink': 'pink',
        }

        found_colors = []
        for cn, en in color_map.items():
            if cn in text.lower():
                found_colors.append(en)
        return list(set(found_colors))

    def _extract_brightness_keywords(self, text: str) -> Dict[str, bool]:
        """从查询文本中提取亮度关键词"""
        bright_words = ['亮', '明亮', '光亮', 'bright', 'light']
        dark_words = ['暗', '黑暗', '昏暗', 'dark', 'dim']

        result = {'bright': False, 'dark': False}
        for word in bright_words:
            if word in text.lower():
                result['bright'] = True
        for word in dark_words:
            if word in text.lower():
                result['dark'] = True
        return result

    def _compute_color_match_score(self, metadata: Dict, target_colors: List[str]) -> float:
        """计算颜色匹配得分"""
        try:
            dominant_colors_str = metadata.get('dominant_colors', '[]')
            if isinstance(dominant_colors_str, str):
                dominant_colors = json.loads(dominant_colors_str)
            else:
                dominant_colors = dominant_colors_str

            if not dominant_colors:
                return 0.5

            # 颜色名称到 BGR 范围的映射
            color_ranges = {
                'red': [(0, 0, 150), (100, 100, 255)],
                'blue': [(150, 0, 0), (255, 100, 100)],
                'green': [(0, 150, 0), (100, 255, 100)],
                'yellow': [(0, 150, 150), (100, 255, 255)],
                'white': [(200, 200, 200), (255, 255, 255)],
                'black': [(0, 0, 0), (50, 50, 50)],
                'orange': [(0, 100, 200), (100, 180, 255)],
                'purple': [(150, 0, 150), (255, 100, 255)],
                'pink': [(150, 150, 200), (255, 200, 255)],
                'gray': [(100, 100, 100), (180, 180, 180)],
                'brown': [(0, 50, 100), (100, 150, 180)],
            }

            max_score = 0
            for target_color in target_colors:
                if target_color not in color_ranges:
                    continue

                low, high = color_ranges[target_color]
                for color in dominant_colors[:3]:  # 只检查前3个主色
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        b, g, r = color[0], color[1], color[2]
                        # 检查是否在颜色范围内
                        if (low[0] <= b <= high[0] and
                            low[1] <= g <= high[1] and
                            low[2] <= r <= high[2]):
                            max_score = max(max_score, 1.0)
                        else:
                            # 计算距离得分
                            dist = sum(abs(color[i] - (low[i] + high[i]) / 2) for i in range(3))
                            score = max(0, 1 - dist / 400)
                            max_score = max(max_score, score)

            return max_score if max_score > 0 else 0.3

        except Exception:
            return 0.5

    def _compute_brightness_match_score(self, brightness: float,
                                        keywords: Dict[str, bool]) -> float:
        """计算亮度匹配得分"""
        if keywords.get('bright') and not keywords.get('dark'):
            # 期望亮图像
            return brightness / 255
        elif keywords.get('dark') and not keywords.get('bright'):
            # 期望暗图像
            return 1 - brightness / 255
        else:
            return 0.5

    def _translate_to_english(self, text: str) -> str:
        """
        将中文翻译为英文（用于CLIP搜索）
        使用简单的词典映射，支持常见物品
        """
        # 检测是否包含中文
        import re
        if not re.search(r'[\u4e00-\u9fff]', text):
            return text  # 非中文直接返回

        # 常见物品中英文映射
        translations = {
            # 饮品容器
            "杯子": "cup mug glass",
            "茶杯": "tea cup",
            "咖啡杯": "coffee cup mug",
            "玻璃杯": "glass",
            "水杯": "water cup glass",
            "酒杯": "wine glass",
            "马克杯": "mug",

            # 餐具
            "碗": "bowl",
            "盘子": "plate dish",
            "筷子": "chopsticks",
            "勺子": "spoon",
            "叉子": "fork",
            "刀": "knife",

            # 食物
            "苹果": "apple",
            "香蕉": "banana",
            "橙子": "orange",
            "水果": "fruit",
            "蔬菜": "vegetable",
            "面包": "bread",
            "蛋糕": "cake",
            "pizza": "pizza",
            "汉堡": "hamburger burger",
            "饭": "rice food meal",
            "面条": "noodles",

            # 动物
            "猫": "cat",
            "狗": "dog",
            "鸟": "bird",
            "鱼": "fish",
            "兔子": "rabbit bunny",
            "马": "horse",
            "牛": "cow cattle",
            "羊": "sheep",
            "猪": "pig",
            "鸡": "chicken",
            "老虎": "tiger",
            "狮子": "lion",
            "熊": "bear",
            "大象": "elephant",
            "猴子": "monkey",

            # 交通工具
            "汽车": "car automobile",
            "车": "car vehicle",
            "自行车": "bicycle bike",
            "摩托车": "motorcycle motorbike",
            "公交车": "bus",
            "火车": "train",
            "飞机": "airplane plane aircraft",
            "船": "boat ship",
            "卡车": "truck",

            # 电子产品
            "手机": "phone mobile smartphone",
            "电脑": "computer laptop PC",
            "电视": "television TV",
            "相机": "camera",
            "耳机": "headphones earphones",
            "键盘": "keyboard",
            "鼠标": "mouse",
            "平板": "tablet iPad",

            # 家具
            "椅子": "chair",
            "桌子": "table desk",
            "沙发": "sofa couch",
            "床": "bed",
            "柜子": "cabinet cupboard",
            "书架": "bookshelf",
            "灯": "lamp light",
            "门": "door",
            "窗户": "window",

            # 服装
            "衣服": "clothes clothing",
            "裤子": "pants trousers",
            "裙子": "dress skirt",
            "鞋": "shoes",
            "帽子": "hat cap",
            "眼镜": "glasses",
            "手表": "watch",
            "包": "bag",
            "背包": "backpack",

            # 自然
            "花": "flower",
            "树": "tree",
            "草": "grass",
            "山": "mountain",
            "水": "water",
            "海": "sea ocean",
            "河": "river",
            "湖": "lake",
            "天空": "sky",
            "云": "cloud",
            "太阳": "sun",
            "月亮": "moon",
            "星星": "star",

            # 人物
            "人": "person people human face portrait",
            "人像": "portrait face headshot photo",
            "大头照": "headshot portrait close-up face",
            "自拍": "selfie portrait face",
            "脸": "face portrait",
            "头像": "portrait avatar headshot",
            "男人": "man male person",
            "女人": "woman female person",
            "孩子": "child kid",
            "婴儿": "baby infant",
            "老人": "elderly old person",

            # 场景
            "房子": "house building",
            "城市": "city urban",
            "街道": "street road",
            "公园": "park",
            "海滩": "beach",
            "森林": "forest",
            "沙漠": "desert",
            "雪": "snow",

            # 颜色（用于风格搜索）
            "红色": "red",
            "蓝色": "blue",
            "绿色": "green",
            "黄色": "yellow",
            "白色": "white",
            "黑色": "black",
            "橙色": "orange",
            "紫色": "purple",
            "粉色": "pink",
            "灰色": "gray grey",

            # 风格
            "复古": "vintage retro",
            "现代": "modern",
            "日落": "sunset",
            "日出": "sunrise",
            "夜景": "night scene",
            "风景": "landscape scenery",
            "肖像": "portrait",
            "美食": "food cuisine",
        }

        # 精确匹配
        if text in translations:
            return translations[text]

        # 部分匹配（检查是否包含某个关键词）
        for cn, en in translations.items():
            if cn in text:
                # 替换中文为英文
                text = text.replace(cn, en)

        # 如果仍有中文，返回原文（可能匹配名称）
        return text

    def search_by_name(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        通过名称/路径关键词搜索图像

        Args:
            query: 搜索关键词
            top_k: 返回数量

        Returns:
            匹配的图像列表
        """
        results = []
        query_lower = query.lower()

        if self.collection is not None:
            try:
                # 获取所有图像并进行名称匹配
                all_images = self.collection.get(include=['metadatas'])

                if all_images and all_images['ids']:
                    for i, image_id in enumerate(all_images['ids']):
                        # 跳过分组标记
                        if image_id.startswith("__group__"):
                            continue

                        metadata = all_images['metadatas'][i]

                        # 跳过空路径
                        path = metadata.get("path", "")
                        if not path:
                            continue

                        # 检查名称匹配
                        name = metadata.get("name", "")
                        filename = Path(path).stem if path else ""

                        # 在名称、文件名、路径中搜索
                        if (query_lower in name.lower() or
                            query_lower in filename.lower() or
                            query_lower in path.lower()):

                            results.append({
                                "id": image_id,
                                "path": path,
                                "similarity": 1.0,  # 名称匹配给高相似度
                                "metadata": metadata
                            })

                            if len(results) >= top_k:
                                break

            except Exception as e:
                print(f"名称搜索失败: {e}")
        else:
            # 内存模式搜索
            for item in self.memory_index:
                # 跳过分组标记
                if item['id'].startswith("__group__"):
                    continue

                metadata = item['metadata']
                path = metadata.get("path", "")

                if not path:
                    continue

                name = metadata.get("name", "")
                filename = Path(path).stem if path else ""

                if (query_lower in name.lower() or
                    query_lower in filename.lower() or
                    query_lower in path.lower()):

                    results.append({
                        "id": item["id"],
                        "path": path,
                        "similarity": 1.0,
                        "metadata": metadata
                    })

                    if len(results) >= top_k:
                        break

        return results

    def get_image_count(self) -> int:
        """获取索引中的图像数量"""
        if self.collection is not None:
            return self.collection.count()
        return len(self.memory_index)

    def get_all_images(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取所有图像列表（支持分页）
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            图像数据列表
        """
        if self.collection is not None:
            # ChromaDB v0.4+ API
            try:
                # 注意: ChromaDB 的 get 方法参数可能通过 limit/offset 控制
                results = self.collection.get(
                    limit=limit,
                    offset=offset,
                    include=['metadatas', 'embeddings']
                )
                
                images = []
                if results and 'ids' in results:
                    ids = results['ids']
                    if ids is not None and len(ids) > 0:
                        for i, image_id in enumerate(ids):
                            metadata = results['metadatas'][i]
                            
                            embedding_val = None
                            embeddings = results.get('embeddings')
                            if embeddings is not None and len(embeddings) > i:
                                embedding_val = embeddings[i]

                            images.append({
                                "id": image_id,
                                "path": metadata.get("path", ""),
                                "metadata": metadata,
                                "embedding": embedding_val
                            })
                return images
            except Exception as e:
                print(f"获取图像列表失败: {e}")
                return []
        else:
            # 内存索引分页
            start = offset
            end = min(offset + limit, len(self.memory_index))
            results = []
            for item in self.memory_index[start:end]:
                results.append({
                    "id": item["id"],
                    "path": item["metadata"].get("path", ""),
                    "metadata": item["metadata"],
                    "embedding": item["embedding"]
                })
            return results

    def remove_image(self, image_id: str):
        """从索引中删除图像"""
        if self.collection is not None:
            self.collection.delete(ids=[image_id])
        else:
            self.memory_index = [item for item in self.memory_index if item["id"] != image_id]

    def update_image_metadata(self, image_id: str, new_metadata: Dict[str, Any]):
        """更新图像元数据"""
        if self.collection is not None:
             try:
                 # 先获取现有metadata
                 res = self.collection.get(ids=[image_id], include=['metadatas'])
                 if res and res['ids']:
                     meta = res['metadatas'][0]
                     meta.update(new_metadata)
                     self.collection.update(ids=[image_id], metadatas=[meta])
             except Exception as e:
                 print(f"更新元数据失败: {e}")
        else:
             for item in self.memory_index:
                 if item['id'] == image_id:
                     item['metadata'].update(new_metadata)
                     break

    def add_group(self, group_name: str):
        """添加一个空分组（持久化）"""
        # 使用一个特殊的元数据记录来保存空分组
        group_id = f"__group__{group_name}"

        if self.collection is not None:
            try:
                # 检查是否已存在
                existing = self.collection.get(ids=[group_id])
                if existing and existing['ids']:
                    return  # 已存在

                # 添加一个特殊的分组记录
                dummy_embedding = [0.0] * self.feature_extractor.embedding_dim
                self.collection.add(
                    ids=[group_id],
                    embeddings=[dummy_embedding],
                    metadatas=[{
                        "path": "",
                        "group": group_name,
                        "__is_group_marker__": "true"
                    }]
                )
            except Exception as e:
                print(f"添加分组失败: {e}")
        else:
            # 内存模式：检查是否已存在
            for item in self.memory_index:
                if item['id'] == group_id:
                    return

            self.memory_index.append({
                "id": group_id,
                "embedding": np.zeros(self.feature_extractor.embedding_dim),
                "metadata": {
                    "path": "",
                    "group": group_name,
                    "__is_group_marker__": "true"
                }
            })

    def delete_group(self, group_name: str):
        """删除分组（将分组内的图片移动到默认分组）"""
        if self.collection is not None:
            try:
                # 1. 先把该分组下的所有图片移到默认分组
                results = self.collection.get(
                    where={"group": group_name},
                    include=['metadatas']
                )

                if results and results['ids']:
                    for i, img_id in enumerate(results['ids']):
                        meta = results['metadatas'][i]
                        # 跳过分组标记
                        if meta.get("__is_group_marker__") == "true":
                            # 删除分组标记
                            self.collection.delete(ids=[img_id])
                        else:
                            # 普通图片移到默认分组
                            meta['group'] = '默认'
                            self.collection.update(ids=[img_id], metadatas=[meta])
            except Exception as e:
                print(f"删除分组失败: {e}")
        else:
            # 内存模式
            for item in self.memory_index:
                if item['metadata'].get('group') == group_name:
                    if item['metadata'].get('__is_group_marker__') == 'true':
                        self.memory_index.remove(item)
                    else:
                        item['metadata']['group'] = '默认'

    def rename_group(self, old_name: str, new_name: str):
        """重命名分组"""
        if self.collection is not None:
            try:
                results = self.collection.get(
                    where={"group": old_name},
                    include=['metadatas']
                )

                if results and results['ids']:
                    for i, img_id in enumerate(results['ids']):
                        meta = results['metadatas'][i]
                        meta['group'] = new_name
                        self.collection.update(ids=[img_id], metadatas=[meta])
            except Exception as e:
                print(f"重命名分组失败: {e}")
        else:
            for item in self.memory_index:
                if item['metadata'].get('group') == old_name:
                    item['metadata']['group'] = new_name

    def get_groups(self) -> List[str]:
        """获取所有分组"""
        groups = set(["默认"])

        if self.collection is not None:
             try:
                results = self.collection.get(include=['metadatas'])
                for meta in results['metadatas']:
                    if 'group' in meta:
                        groups.add(meta['group'])
             except Exception:
                pass
        else:
            for item in self.memory_index:
                if 'group' in item['metadata']:
                    groups.add(item['metadata']['group'])

        return sorted(list(groups))

    def get_images_by_group(self, group: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """按分组获取图像（排除分组标记）"""
        if self.collection is not None:
            # 使用 where 过滤
            where_filter = {"group": group} if group != "全部" else None
            try:
                results = self.collection.get(
                    limit=limit + 10,  # 多取一些以补偿被过滤的标记
                    offset=offset,
                    where=where_filter,
                    include=['metadatas', 'embeddings']
                )

                images = []
                if results and 'ids' in results:
                    ids = results['ids']
                    if ids is not None and len(ids) > 0:
                        for i, image_id in enumerate(ids):
                            metadata = results['metadatas'][i]

                            # 跳过分组标记
                            if metadata.get("__is_group_marker__") == "true":
                                continue

                            # 跳过空路径
                            if not metadata.get("path"):
                                continue

                            embedding_val = None
                            embeddings = results.get('embeddings')
                            if embeddings is not None and len(embeddings) > i:
                                embedding_val = embeddings[i]

                            images.append({
                                "id": image_id,
                                "path": metadata.get("path", ""),
                                "metadata": metadata,
                                "embedding": embedding_val
                            })

                            if len(images) >= limit:
                                break
                return images
            except Exception as e:
                print(f"分组获取失败: {e}")
                return []
        else:
            # 内存过滤
            filtered = [
                item for item in self.memory_index
                if (group == "全部" or item['metadata'].get('group', '默认') == group)
                and item['metadata'].get('__is_group_marker__') != 'true'
                and item['metadata'].get('path')
            ]
            start = offset
            end = min(offset + limit, len(filtered))
            results = []
            for item in filtered[start:end]:
                results.append({
                    "id": item["id"],
                    "path": item["metadata"].get("path", ""),
                    "metadata": item["metadata"],
                    "embedding": item["embedding"]
                })
            return results
