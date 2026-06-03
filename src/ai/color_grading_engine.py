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

        # 1. 曝光与对比度 (BGR Float32)
        img = self._adjust_exposure(img, params.exposure)
        img = self._adjust_contrast(img, params.contrast)

        # 2. 白色与黑色 (采用渐进平滑插值过渡)
        img = self._adjust_whites_blacks(img, params.whites, params.blacks)

        # 3. 色温和色调 (合并 LAB 转换)
        if params.temperature != 0 or params.tint != 0:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
            if params.tint != 0:
                lab[:, :, 1] = np.clip(lab[:, :, 1] + params.tint * 1.28, 0, 255)
            if params.temperature != 0:
                lab[:, :, 2] = np.clip(lab[:, :, 2] + params.temperature * 1.28, 0, 255)
            img_uint8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            img = img_uint8.astype(np.float32) / 255.0

        # 4. 高光、阴影、饱和度、自然饱和度、色相偏移 (合并 HSV 转换)
        has_hsv = (params.highlights != 0 or params.shadows != 0 or
                   params.saturation != 1.0 or params.vibrance != 0 or
                   params.hue_shift != 0)
        if has_hsv:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

            # 高光 & 阴影
            if params.highlights != 0 or params.shadows != 0:
                v = hsv[:, :, 2] / 255.0
                if params.highlights != 0:
                    highlight_mask = np.clip((v - 0.5) * 2.0, 0, 1)
                    v = v + highlight_mask * (params.highlights / 100.0) * 0.3
                if params.shadows != 0:
                    shadow_mask = np.clip((0.5 - v) * 2.0, 0, 1)
                    v = v + shadow_mask * (params.shadows / 100.0) * 0.3
                hsv[:, :, 2] = np.clip(v * 255.0, 0, 255)

            # 饱和度 & 自然饱和度
            if params.saturation != 1.0 or params.vibrance != 0:
                s = hsv[:, :, 1]
                if params.saturation != 1.0:
                    s = s * params.saturation
                if params.vibrance != 0:
                    sat_factor = 1.0 - (s / 255.0)
                    s = s + sat_factor * (params.vibrance / 100.0) * 50.0
                hsv[:, :, 1] = np.clip(s, 0, 255)

            # 色相偏移
            if params.hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + params.hue_shift / 2) % 180

            img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img = img_uint8.astype(np.float32) / 255.0

        # 5. 清晰度 (直接在 float32 运算)
        if params.clarity != 0:
            img = self._adjust_clarity(img, params.clarity)

        # 6. 去雾 (直接在 float32 运算)
        if params.dehaze != 0:
            img = self._dehaze(img, params.dehaze)

        # 7. 分离色调 (直接在 float32 运算，使用 broadcasting)
        if params.split_tone_shadows != [0, 0, 0] or params.split_tone_highlights != [255, 255, 255]:
            img = self._apply_split_toning(img, params.split_tone_shadows,
                                          params.split_tone_highlights, params.split_tone_balance)

        # 8. 褪色效果
        if params.fade > 0:
            img = self._apply_fade(img, params.fade)

        # 9. 暗角
        if params.vignette > 0:
            img = self._apply_vignette(img, params.vignette)

        # 10. 颗粒
        if params.grain > 0:
            img = self._apply_grain(img, params.grain)

        # 裁剪并转换回uint8
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 11. 自定义曲线 (直接在 uint8 上执行，极速)
        if params.tone_curve:
            img_uint8 = self._apply_tone_curve_uint8(img_uint8, params.tone_curve)

        return img_uint8

    def _adjust_exposure(self, img: np.ndarray, exposure: float) -> np.ndarray:
        """曝光调整"""
        if exposure == 0:
            return img
        # 使用2的幂次方调整,模拟相机曝光
        factor = 2.0 ** exposure
        return img * factor

    def _adjust_white_balance(self, img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
        """色温和色调调整 (兼容层：直接调用优化管道)"""
        # 如果需要色温和色调，通过单独的快转换完成
        if temperature == 0 and tint == 0:
            return img
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)
        if tint != 0:
            lab[:, :, 1] = np.clip(lab[:, :, 1] + tint * 1.28, 0, 255)
        if temperature != 0:
            lab[:, :, 2] = np.clip(lab[:, :, 2] + temperature * 1.28, 0, 255)
        img_uint8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return img_uint8.astype(np.float32) / 255.0

    def _adjust_contrast(self, img: np.ndarray, contrast: float) -> np.ndarray:
        """对比度调整"""
        if contrast == 1.0:
            return img
        mean = 0.5
        return (img - mean) * contrast + mean

    def _adjust_highlights_shadows(self, img: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
        """高光和阴影调整 (兼容层：直接调用优化快转换)"""
        if highlights == 0 and shadows == 0:
            return img
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0
        if highlights != 0:
            highlight_mask = np.clip((v - 0.5) * 2.0, 0, 1)
            v = v + highlight_mask * (highlights / 100.0) * 0.3
        if shadows != 0:
            shadow_mask = np.clip((0.5 - v) * 2.0, 0, 1)
            v = v + shadow_mask * (shadows / 100.0) * 0.3
        hsv[:, :, 2] = np.clip(v * 255.0, 0, 255)
        img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_uint8.astype(np.float32) / 255.0

    def _adjust_whites_blacks(self, img: np.ndarray, whites: float, blacks: float) -> np.ndarray:
        """白色和黑色调整 (采用渐变平滑过渡，杜绝色彩断带)"""
        if whites == 0 and blacks == 0:
            return img

        result = img.copy()

        # 对于 whites (高光亮部)：从 0.7 到 1.0 建立平滑插值遮罩
        if whites != 0:
            mask = np.clip((result - 0.7) / 0.3, 0.0, 1.0)
            smooth_mask = mask * mask * (3.0 - 2.0 * mask)  # Smoothstep
            white_point = 1.0 + whites / 200.0
            adjusted = result * white_point
            result = result * (1.0 - smooth_mask) + adjusted * smooth_mask

        # 对于 blacks (暗部深色部)：从 0.0 到 0.3 建立平滑插值遮罩
        if blacks != 0:
            mask = np.clip((0.3 - result) / 0.3, 0.0, 1.0)
            smooth_mask = mask * mask * (3.0 - 2.0 * mask)  # Smoothstep
            black_lift = blacks / 200.0
            adjusted = result + black_lift
            result = result * (1.0 - smooth_mask) + adjusted * smooth_mask

        return result

    def _adjust_saturation(self, img: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
        """饱和度和自然饱和度调整 (兼容层)"""
        if saturation == 1.0 and vibrance == 0:
            return img
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        s = hsv[:, :, 1]
        if saturation != 1.0:
            s = s * saturation
        if vibrance != 0:
            sat_factor = 1.0 - (s / 255.0)
            s = s + sat_factor * (vibrance / 100.0) * 50.0
        hsv[:, :, 1] = np.clip(s, 0, 255)
        img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_uint8.astype(np.float32) / 255.0

    def _adjust_hue(self, img: np.ndarray, hue_shift: float) -> np.ndarray:
        """色相偏移 (兼容层)"""
        if hue_shift == 0:
            return img
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2) % 180
        img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_uint8.astype(np.float32) / 255.0

    def _adjust_clarity(self, img: np.ndarray, clarity: float) -> np.ndarray:
        """清晰度调整 (中频对比度) - 直接在 Float32 空间上运算"""
        if clarity == 0:
            return img
        blur = cv2.GaussianBlur(img, (0, 0), 10)
        factor = 1.0 + clarity / 100.0
        return cv2.addWeighted(img, factor, blur, 1.0 - factor, 0.0)

    def _dehaze(self, img: np.ndarray, strength: float) -> np.ndarray:
        """去雾效果 - 直接在 Float32 空间上运算"""
        if strength == 0:
            return img

        # 估计大气光
        dark_channel = np.min(img, axis=2)
        atmospheric_light = np.percentile(dark_channel, 99)

        # 去雾
        factor = strength / 100.0
        result = (img - atmospheric_light * factor) / (1.0 - factor + 1e-6) + atmospheric_light * factor
        return result

    def _apply_split_toning(self, img: np.ndarray, shadows_color: list,
                            highlights_color: list, balance: float) -> np.ndarray:
        """分离色调 - 直接在 Float32 空间上运算，采用 broadcasting 加速"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 创建阴影和高光遮罩
        shadow_mask = np.clip(1.0 - gray * 2.0, 0.0, 1.0)
        highlight_mask = np.clip(gray * 2.0 - 1.0, 0.0, 1.0)

        # 应用平衡
        balance_factor = (balance + 100.0) / 200.0
        shadow_mask *= (1.0 - balance_factor)
        highlight_mask *= balance_factor

        result = img.copy()
        shadows_bgr = np.array(shadows_color[::-1]) / 255.0  # BGR
        highlights_bgr = np.array(highlights_color[::-1]) / 255.0

        # 使用 broadcasting 避免 for 循环 (提升通道计算效率)
        shadow_mask_expanded = shadow_mask[:, :, np.newaxis]
        highlight_mask_expanded = highlight_mask[:, :, np.newaxis]

        result = result * (1.0 - shadow_mask_expanded * 0.3) + shadows_bgr * shadow_mask_expanded * 0.3
        result = result * (1.0 - highlight_mask_expanded * 0.3) + highlights_bgr * highlight_mask_expanded * 0.3

        return result

    def _apply_fade(self, img: np.ndarray, fade: float) -> np.ndarray:
        """褪色效果"""
        if fade <= 0:
            return img

        # 提升黑色点
        black_point = fade * 0.3
        result = img * (1.0 - fade) + black_point + img * fade * 0.7
        return np.clip(result, 0, 1.0)

    def _apply_vignette(self, img: np.ndarray, strength: float) -> np.ndarray:
        """暗角效果"""
        if strength <= 0:
            return img

        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0

        # 计算到中心的距离
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        # 创建暗角遮罩
        vignette = 1.0 - (dist / max_dist) ** 2 * (strength / 100.0)
        vignette = np.clip(vignette, 0, 1.0)

        return img * vignette[:, :, np.newaxis]

    def _apply_grain(self, img: np.ndarray, amount: float) -> np.ndarray:
        """颗粒效果"""
        if amount <= 0:
            return img

        noise = np.random.normal(0, amount / 100.0 * 0.1, img.shape)
        result = img + noise
        return np.clip(result, 0, 1.0)

    def _apply_tone_curve_uint8(self, img_uint8: np.ndarray, curve_points: list) -> np.ndarray:
        """直接在 uint8 图像上应用自定义曲线查找表"""
        if not curve_points or len(curve_points) < 2:
            return img_uint8

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

        return cv2.LUT(img_uint8, lut)

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
