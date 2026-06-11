"""
调色引擎
基于端侧低bit量化技术实现高效图像调色
"""
import colorsys
import numpy as np
from typing import Optional, Tuple, Dict, Any
import cv2
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from .color_params import ColorGradingParams


class ColorGradingEngine:
    """
    调色引擎
    支持基于参数的调色和基于AI模型的智能调色
    """

    HSL_CHANNELS = (
        ("red", 0.0, 34.0),
        ("orange", 30.0, 26.0),
        ("yellow", 60.0, 30.0),
        ("green", 120.0, 48.0),
        ("aqua", 180.0, 34.0),
        ("blue", 235.0, 44.0),
        ("purple", 275.0, 34.0),
        ("magenta", 315.0, 36.0),
    )

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

        # 1. 基础明暗与通道级校正 (BGR Float32)
        img = self._adjust_exposure(img, params.exposure)
        img = self._adjust_brightness(img, params.brightness)
        img = self._adjust_contrast(img, params.contrast)
        img = self._adjust_gamma(img, params.gamma)
        img = self._apply_channel_balance(img, params)
        img = self._apply_cdl(img, params)

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

        # 4. 高光、阴影、饱和度、自然饱和度、色相偏移、HSL 分色 (合并 HSV 转换)
        has_hsv = (params.highlights != 0 or params.shadows != 0 or
                   params.saturation != 1.0 or params.vibrance != 0 or
                   params.hue_shift != 0 or self._has_hsl_adjustments(params))
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

            if self._has_hsl_adjustments(params):
                hsv = self._apply_hsl_color_mix(hsv, params)

            img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img = img_uint8.astype(np.float32) / 255.0

        # 5. 三路色轮 / 分离色调
        if self._has_tonal_color_wheels(params):
            img = self._apply_tonal_color_wheels(img, params)

        # 6. 清晰度、纹理与中间调细节 (直接在 float32 运算)
        if params.texture != 0:
            img = self._adjust_texture(img, params.texture)

        if params.midtone_detail != 0:
            img = self._adjust_midtone_detail(img, params.midtone_detail)

        if params.clarity != 0:
            img = self._adjust_clarity(img, params.clarity)

        if params.noise_reduction > 0:
            img = self._apply_noise_reduction(img, params.noise_reduction)

        if params.sharpen > 0:
            img = self._apply_sharpen(img, params.sharpen)

        # 7. 去雾 (直接在 float32 运算)
        if params.dehaze != 0:
            img = self._dehaze(img, params.dehaze)

        if params.bloom > 0:
            img = self._apply_bloom(img, params.bloom)

        # 8. 分离色调 (直接在 float32 运算，使用 broadcasting)
        if params.split_tone_shadows != [0, 0, 0] or params.split_tone_highlights != [255, 255, 255]:
            img = self._apply_split_toning(img, params.split_tone_shadows,
                                          params.split_tone_highlights, params.split_tone_balance)

        # 9. 参数曲线
        if self._has_parametric_curve(params):
            img = self._apply_parametric_curve(img, params)

        # 10. 褪色效果
        if params.fade > 0:
            img = self._apply_fade(img, params.fade)

        # 11. 暗角
        if params.vignette > 0:
            img = self._apply_vignette(img, params.vignette)

        # 12. 颗粒
        if params.grain > 0:
            img = self._apply_grain(img, params.grain)

        # 裁剪并转换回uint8
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # 13. 自定义曲线 (直接在 uint8 上执行，极速)
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

    def _adjust_brightness(self, img: np.ndarray, brightness: float) -> np.ndarray:
        """亮度调整"""
        if brightness == 0:
            return img
        return img + brightness

    def _adjust_gamma(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma / 中间调亮度调整。gamma > 1 提亮中间调，gamma < 1 压暗。"""
        if gamma == 1.0:
            return img
        safe_gamma = max(0.05, float(gamma))
        return np.power(np.clip(img, 0.0, 1.0), 1.0 / safe_gamma)

    def _apply_channel_balance(self, img: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """RGB 通道平衡，适合轻微校色或风格化偏色。"""
        if params.red_balance == 0 and params.green_balance == 0 and params.blue_balance == 0:
            return img
        factors = np.array([
            1.0 + params.blue_balance / 200.0,
            1.0 + params.green_balance / 200.0,
            1.0 + params.red_balance / 200.0,
        ], dtype=np.float32)
        return img * np.clip(factors, 0.0, 3.0)

    def _apply_cdl(self, img: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """应用 ASC-CDL 风格的 slope / offset / power / saturation 调整。"""
        if (self._is_default_rgb_list(params.cdl_slope, 1.0) and
                self._is_default_rgb_list(params.cdl_offset, 0.0) and
                self._is_default_rgb_list(params.cdl_power, 1.0) and
                params.cdl_saturation == 1.0):
            return img

        slope = self._rgb_list_to_bgr_array(params.cdl_slope, 1.0, min_value=0.0)
        offset = self._rgb_list_to_bgr_array(params.cdl_offset, 0.0)
        power = self._rgb_list_to_bgr_array(params.cdl_power, 1.0, min_value=0.05)

        result = np.power(np.clip(img * slope + offset, 0.0, None), power)
        if params.cdl_saturation != 1.0:
            result = self._adjust_rgb_saturation_float(result, params.cdl_saturation)
        return result

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

    def _has_hsl_adjustments(self, params: ColorGradingParams) -> bool:
        """是否存在单色 HSL 分色调整。"""
        for color_name, _, _ in self.HSL_CHANNELS:
            if (getattr(params, f"{color_name}_hue") != 0 or
                    getattr(params, f"{color_name}_saturation") != 0 or
                    getattr(params, f"{color_name}_luminance") != 0):
                return True
        return False

    def _apply_hsl_color_mix(self, hsv: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """按颜色范围调整 HSL，用于天空、肤色、草地等语义目标。"""
        hue_degrees = hsv[:, :, 0] * 2.0
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        for color_name, center, width in self.HSL_CHANNELS:
            hue_shift = getattr(params, f"{color_name}_hue")
            saturation = getattr(params, f"{color_name}_saturation")
            luminance = getattr(params, f"{color_name}_luminance")
            if hue_shift == 0 and saturation == 0 and luminance == 0:
                continue

            distance = np.abs((hue_degrees - center + 180.0) % 360.0 - 180.0)
            mask = np.clip(1.0 - distance / width, 0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)

            if hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + (hue_shift / 2.0) * mask) % 180.0
                hue_degrees = hsv[:, :, 0] * 2.0
            if saturation != 0:
                sat = sat * (1.0 + mask * saturation / 100.0)
            if luminance != 0:
                val = val + mask * (luminance / 100.0) * 64.0

        hsv[:, :, 1] = np.clip(sat, 0, 255)
        hsv[:, :, 2] = np.clip(val, 0, 255)
        return hsv

    def _has_tonal_color_wheels(self, params: ColorGradingParams) -> bool:
        """是否存在阴影/中间调/高光色轮调整。"""
        return (
            params.shadow_saturation > 0 or
            params.midtone_saturation > 0 or
            params.highlight_saturation > 0
        )

    def _apply_tonal_color_wheels(self, img: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """模拟三路色轮，将色彩分别注入阴影、中间调和高光。"""
        safe_img = np.clip(img, 0.0, 1.0).astype(np.float32)
        gray = cv2.cvtColor(safe_img, cv2.COLOR_BGR2GRAY)

        shadow_mask = np.clip((0.55 - gray) / 0.55, 0.0, 1.0)
        shadow_mask = shadow_mask * shadow_mask * (3.0 - 2.0 * shadow_mask)

        highlight_mask = np.clip((gray - 0.45) / 0.55, 0.0, 1.0)
        highlight_mask = highlight_mask * highlight_mask * (3.0 - 2.0 * highlight_mask)

        midtone_mask = np.clip(1.0 - np.abs(gray - 0.5) / 0.35, 0.0, 1.0)
        midtone_mask = midtone_mask * midtone_mask * (3.0 - 2.0 * midtone_mask)

        result = safe_img.copy()
        result = self._blend_tonal_tint(result, shadow_mask, params.shadow_hue, params.shadow_saturation)
        result = self._blend_tonal_tint(result, midtone_mask, params.midtone_hue, params.midtone_saturation)
        result = self._blend_tonal_tint(result, highlight_mask, params.highlight_hue, params.highlight_saturation)
        return result.astype(np.float32)

    def _blend_tonal_tint(
            self,
            img: np.ndarray,
            mask: np.ndarray,
            hue: float,
            saturation: float) -> np.ndarray:
        if saturation <= 0:
            return img
        tint = self._hue_to_bgr(hue)
        amount = np.clip(saturation / 100.0, 0.0, 1.0) * 0.35
        blend = mask[:, :, np.newaxis] * amount
        return (img * (1.0 - blend) + tint * blend).astype(np.float32)

    def _adjust_clarity(self, img: np.ndarray, clarity: float) -> np.ndarray:
        """清晰度调整 (中频对比度) - 直接在 Float32 空间上运算"""
        if clarity == 0:
            return img
        blur = cv2.GaussianBlur(img, (0, 0), 10)
        factor = 1.0 + clarity / 100.0
        return cv2.addWeighted(img, factor, blur, 1.0 - factor, 0.0)

    def _adjust_texture(self, img: np.ndarray, texture: float) -> np.ndarray:
        """纹理调整，主要影响细小边缘。负值会柔化皮肤和细节。"""
        if texture == 0:
            return img
        blur = cv2.GaussianBlur(img, (0, 0), 1.4)
        detail = img - blur
        return img + detail * (texture / 100.0) * 0.7

    def _adjust_midtone_detail(self, img: np.ndarray, detail_amount: float) -> np.ndarray:
        """中间调细节，只在中间亮度范围增强或压低局部反差。"""
        if detail_amount == 0:
            return img
        safe_img = np.clip(img, 0.0, 1.0).astype(np.float32)
        gray = cv2.cvtColor(safe_img, cv2.COLOR_BGR2GRAY)
        mask = np.clip(1.0 - np.abs(gray - 0.5) / 0.35, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        blur = cv2.GaussianBlur(safe_img, (0, 0), 6)
        detail = safe_img - blur
        return (safe_img + detail * mask[:, :, np.newaxis] * (detail_amount / 100.0) * 1.2).astype(np.float32)

    def _apply_sharpen(self, img: np.ndarray, strength: float) -> np.ndarray:
        """锐化。"""
        if strength <= 0:
            return img
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        return img + (img - blur) * np.clip(strength / 100.0, 0.0, 1.0) * 0.9

    def _apply_noise_reduction(self, img: np.ndarray, strength: float) -> np.ndarray:
        """轻量降噪，使用双边滤波保留主要边缘。"""
        if strength <= 0:
            return img
        safe_img = np.clip(img, 0.0, 1.0).astype(np.float32)
        sigma_color = 0.03 + np.clip(strength / 100.0, 0.0, 1.0) * 0.18
        sigma_space = 3.0 + np.clip(strength / 100.0, 0.0, 1.0) * 8.0
        smoothed = cv2.bilateralFilter(safe_img, 5, sigma_color, sigma_space)
        amount = np.clip(strength / 100.0, 0.0, 1.0)
        return img * (1.0 - amount) + smoothed * amount

    def _apply_bloom(self, img: np.ndarray, strength: float) -> np.ndarray:
        """高光柔光 / 泛光。"""
        if strength <= 0:
            return img
        safe_img = np.clip(img, 0.0, 1.0).astype(np.float32)
        gray = cv2.cvtColor(safe_img, cv2.COLOR_BGR2GRAY)
        mask = np.clip((gray - 0.62) / 0.38, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        glow = safe_img * mask[:, :, np.newaxis]
        sigma = 8.0 + np.clip(strength, 0.0, 100.0) * 0.18
        glow = cv2.GaussianBlur(glow, (0, 0), sigma)
        return (safe_img + glow * np.clip(strength / 100.0, 0.0, 1.0) * 0.8).astype(np.float32)

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
        safe_img = np.clip(img, 0.0, 1.0).astype(np.float32)
        gray = cv2.cvtColor(safe_img, cv2.COLOR_BGR2GRAY)

        # 创建阴影和高光遮罩
        shadow_mask = np.clip(1.0 - gray * 2.0, 0.0, 1.0)
        highlight_mask = np.clip(gray * 2.0 - 1.0, 0.0, 1.0)

        # 应用平衡
        balance_factor = (balance + 100.0) / 200.0
        shadow_mask *= (1.0 - balance_factor)
        highlight_mask *= balance_factor

        result = safe_img.copy()
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

    def _has_parametric_curve(self, params: ColorGradingParams) -> bool:
        """是否存在参数曲线调整。"""
        return (
            params.curve_shadows != 0 or
            params.curve_darks != 0 or
            params.curve_lights != 0 or
            params.curve_highlights != 0
        )

    def _apply_parametric_curve(self, img: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """Lightroom 风格的参数曲线，用四个亮度区域塑造影调。"""
        result = np.clip(img, 0.0, 1.0).copy()
        zones = (
            (params.curve_shadows, 0.12, 0.26),
            (params.curve_darks, 0.34, 0.34),
            (params.curve_lights, 0.66, 0.34),
            (params.curve_highlights, 0.88, 0.26),
        )
        for value, center, width in zones:
            if value == 0:
                continue
            distance = np.abs(result - center) / width
            mask = np.clip(1.0 - distance, 0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)
            result = result + mask * (value / 100.0) * 0.22
        return np.clip(result, 0.0, 1.0)

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

    def _hue_to_bgr(self, hue: float) -> np.ndarray:
        """将 0-360 色相转换为 BGR Float32 颜色。"""
        r, g, b = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, 1.0)
        return np.array([b, g, r], dtype=np.float32)

    def _rgb_list_to_bgr_array(
            self,
            value: list,
            default: float,
            min_value: Optional[float] = None) -> np.ndarray:
        """将 RGB 列表转换为适合 BGR 图像相乘的数组。"""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rgb = [default, default, default]
        else:
            rgb = [float(item) for item in value]
        if min_value is not None:
            rgb = [max(min_value, item) for item in rgb]
        return np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)

    def _is_default_rgb_list(self, value: list, default: float) -> bool:
        """判断 RGB 列表是否为默认值。"""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return True
        return all(abs(float(item) - default) < 1e-6 for item in value)

    def _adjust_rgb_saturation_float(self, img: np.ndarray, saturation: float) -> np.ndarray:
        """按照 Rec.709 亮度权重调整 BGR Float32 图像饱和度。"""
        weights = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
        luma = np.sum(img * weights, axis=2, keepdims=True)
        return luma + saturation * (img - luma)

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
