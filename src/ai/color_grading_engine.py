"""
调色引擎
基于端侧低bit量化技术实现高效图像调色
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
import cv2
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .nlp_color_parser import ColorGradingParams


class ColorGradingEngine:
    """
    调色引擎
    支持基于参数的调色和基于AI模型的智能调色
    """

    def __init__(self, model_path: Optional[Path] = None, use_gpu: bool = True):
        """
        初始化调色引擎

        Args:
            model_path: ONNX模型路径(可选)
            use_gpu: 是否使用GPU加速
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None

        if model_path and model_path.exists() and ort:
            self._load_model()

    def _load_model(self):
        """加载ONNX模型"""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.session = None

    def apply_grading(self, image: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """
        应用调色参数到图像

        Args:
            image: 输入图像 (BGR格式, uint8)
            params: 调色参数

        Returns:
            调色后的图像
        """
        # 转换为浮点数进行处理
        img = image.astype(np.float32) / 255.0

        # 1. 曝光调整
        img = self._adjust_exposure(img, params.exposure)

        # 2. 色温和色调
        img = self._adjust_white_balance(img, params.temperature, params.tint)

        # 3. 对比度
        img = self._adjust_contrast(img, params.contrast)

        # 4. 高光和阴影
        img = self._adjust_highlights_shadows(img, params.highlights, params.shadows)

        # 5. 白色和黑色
        img = self._adjust_whites_blacks(img, params.whites, params.blacks)

        # 6. 饱和度和自然饱和度
        img = self._adjust_saturation(img, params.saturation, params.vibrance)

        # 7. 色相偏移
        if params.hue_shift != 0:
            img = self._adjust_hue(img, params.hue_shift)

        # 8. 清晰度
        if params.clarity != 0:
            img = self._adjust_clarity(img, params.clarity)

        # 9. 去雾
        if params.dehaze != 0:
            img = self._dehaze(img, params.dehaze)

        # 10. 分离色调
        if params.split_tone_shadows != [0, 0, 0] or params.split_tone_highlights != [255, 255, 255]:
            img = self._apply_split_toning(img, params.split_tone_shadows,
                                          params.split_tone_highlights, params.split_tone_balance)

        # 11. 褪色效果
        if params.fade > 0:
            img = self._apply_fade(img, params.fade)

        # 12. 暗角
        if params.vignette > 0:
            img = self._apply_vignette(img, params.vignette)

        # 13. 颗粒
        if params.grain > 0:
            img = self._apply_grain(img, params.grain)

        # 14. 自定义曲线
        if params.tone_curve:
            img = self._apply_tone_curve(img, params.tone_curve)

        # 裁剪并转换回uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def _adjust_exposure(self, img: np.ndarray, exposure: float) -> np.ndarray:
        """曝光调整"""
        if exposure == 0:
            return img
        # 使用2的幂次方调整,模拟相机曝光
        factor = 2 ** exposure
        return img * factor

    def _adjust_white_balance(self, img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
        """色温和色调调整"""
        if temperature == 0 and tint == 0:
            return img

        # 转换到LAB色彩空间
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 调整a通道(绿-品红) - tint
        lab[:, :, 1] = lab[:, :, 1] + tint * 1.28

        # 调整b通道(蓝-黄) - temperature
        lab[:, :, 2] = lab[:, :, 2] + temperature * 1.28

        lab = np.clip(lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result.astype(np.float32) / 255.0

    def _adjust_contrast(self, img: np.ndarray, contrast: float) -> np.ndarray:
        """对比度调整"""
        if contrast == 1.0:
            return img
        mean = 0.5
        return (img - mean) * contrast + mean

    def _adjust_highlights_shadows(self, img: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
        """高光和阴影调整"""
        if highlights == 0 and shadows == 0:
            return img

        # 转换到HSV
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

        v = hsv[:, :, 2] / 255.0

        # 高光调整 (影响亮部)
        if highlights != 0:
            highlight_mask = np.clip((v - 0.5) * 2, 0, 1)
            v = v + highlight_mask * (highlights / 100.0) * 0.3

        # 阴影调整 (影响暗部)
        if shadows != 0:
            shadow_mask = np.clip((0.5 - v) * 2, 0, 1)
            v = v + shadow_mask * (shadows / 100.0) * 0.3

        hsv[:, :, 2] = np.clip(v * 255, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result.astype(np.float32) / 255.0

    def _adjust_whites_blacks(self, img: np.ndarray, whites: float, blacks: float) -> np.ndarray:
        """白色和黑色调整"""
        if whites == 0 and blacks == 0:
            return img

        result = img.copy()

        if whites != 0:
            # 白色调整 - 影响最亮的部分
            white_point = 1.0 + whites / 200.0
            result = np.where(result > 0.9, result * white_point, result)

        if blacks != 0:
            # 黑色调整 - 影响最暗的部分
            black_lift = blacks / 200.0
            result = np.where(result < 0.1, result + black_lift, result)

        return result

    def _adjust_saturation(self, img: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
        """饱和度和自然饱和度调整"""
        if saturation == 1.0 and vibrance == 0:
            return img

        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

        s = hsv[:, :, 1]

        # 饱和度调整
        if saturation != 1.0:
            s = s * saturation

        # 自然饱和度调整 (对低饱和度区域影响更大)
        if vibrance != 0:
            sat_factor = 1.0 - (s / 255.0)  # 低饱和度区域权重更高
            s = s + sat_factor * (vibrance / 100.0) * 50

        hsv[:, :, 1] = np.clip(s, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result.astype(np.float32) / 255.0

    def _adjust_hue(self, img: np.ndarray, hue_shift: float) -> np.ndarray:
        """色相偏移"""
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

        # OpenCV的H范围是0-179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2) % 180

        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result.astype(np.float32) / 255.0

    def _adjust_clarity(self, img: np.ndarray, clarity: float) -> np.ndarray:
        """清晰度调整 (中频对比度)"""
        if clarity == 0:
            return img

        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 使用unsharp mask技术
        blur = cv2.GaussianBlur(img_uint8, (0, 0), 10)
        factor = 1.0 + clarity / 100.0

        result = cv2.addWeighted(img_uint8, factor, blur, 1 - factor, 0)
        return result.astype(np.float32) / 255.0

    def _dehaze(self, img: np.ndarray, strength: float) -> np.ndarray:
        """去雾效果"""
        if strength == 0:
            return img

        # 简化的去雾算法
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

        # 估计大气光
        dark_channel = np.min(img_uint8, axis=2)
        atmospheric_light = np.percentile(dark_channel, 99)

        # 去雾
        factor = strength / 100.0
        result = img_uint8.astype(np.float32)
        result = (result - atmospheric_light * factor) / (1 - factor) + atmospheric_light * factor

        return np.clip(result, 0, 255).astype(np.uint8).astype(np.float32) / 255.0

    def _apply_split_toning(self, img: np.ndarray, shadows_color: list,
                            highlights_color: list, balance: float) -> np.ndarray:
        """分离色调"""
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 创建阴影和高光遮罩
        shadow_mask = np.clip(1.0 - gray * 2, 0, 1)
        highlight_mask = np.clip(gray * 2 - 1, 0, 1)

        # 应用平衡
        balance_factor = (balance + 100) / 200.0
        shadow_mask *= (1 - balance_factor)
        highlight_mask *= balance_factor

        result = img.copy()
        shadows_rgb = np.array(shadows_color[::-1]) / 255.0  # BGR
        highlights_rgb = np.array(highlights_color[::-1]) / 255.0

        for i in range(3):
            result[:, :, i] = result[:, :, i] * (1 - shadow_mask * 0.3) + shadows_rgb[i] * shadow_mask * 0.3
            result[:, :, i] = result[:, :, i] * (1 - highlight_mask * 0.3) + highlights_rgb[i] * highlight_mask * 0.3

        return result

    def _apply_fade(self, img: np.ndarray, fade: float) -> np.ndarray:
        """褪色效果"""
        if fade <= 0:
            return img

        # 提升黑色点
        black_point = fade * 0.3
        result = img * (1 - fade) + black_point + img * fade * 0.7
        return np.clip(result, 0, 1)

    def _apply_vignette(self, img: np.ndarray, strength: float) -> np.ndarray:
        """暗角效果"""
        if strength <= 0:
            return img

        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # 计算到中心的距离
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        # 创建暗角遮罩
        vignette = 1 - (dist / max_dist) ** 2 * (strength / 100.0)
        vignette = np.clip(vignette, 0, 1)

        return img * vignette[:, :, np.newaxis]

    def _apply_grain(self, img: np.ndarray, amount: float) -> np.ndarray:
        """颗粒效果"""
        if amount <= 0:
            return img

        noise = np.random.normal(0, amount / 100.0 * 0.1, img.shape)
        result = img + noise
        return np.clip(result, 0, 1)

    def _apply_tone_curve(self, img: np.ndarray, curve_points: list) -> np.ndarray:
        """应用自定义曲线"""
        if not curve_points or len(curve_points) < 2:
            return img

        # 创建查找表
        lut = np.zeros(256, dtype=np.uint8)
        points = sorted(curve_points, key=lambda x: x[0])

        for i in range(256):
            # 找到i所在的区间
            for j in range(len(points) - 1):
                if points[j][0] <= i <= points[j + 1][0]:
                    # 线性插值
                    t = (i - points[j][0]) / (points[j + 1][0] - points[j][0])
                    lut[i] = int(points[j][1] + t * (points[j + 1][1] - points[j][1]))
                    break
            else:
                lut[i] = i

        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        result = cv2.LUT(img_uint8, lut)
        return result.astype(np.float32) / 255.0

    def extract_color_params(self, image: np.ndarray) -> ColorGradingParams:
        """
        从图像中提取调色参数特征
        用于"复刻"功能

        Args:
            image: 输入图像

        Returns:
            估计的调色参数
        """
        params = ColorGradingParams()

        img = image.astype(np.float32) / 255.0

        # 分析曝光
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 255.0
        params.exposure = (mean_brightness - 0.5) * 2

        # 分析色温 (通过蓝黄比)
        b, g, r = cv2.split(image.astype(np.float32))
        blue_yellow_ratio = np.mean(b) / (np.mean(r) + 1e-6)
        params.temperature = (1 - blue_yellow_ratio) * 50

        # 分析饱和度
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        params.saturation = np.mean(hsv[:, :, 1]) / 127.5

        # 分析对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        params.contrast = np.std(gray) / 64.0

        return params
