"""
AI模块初始化
使用延迟导入优化启动速度
"""

# 延迟导入函数
def __getattr__(name):
    """延迟导入模块成员"""
    if name == 'NLPColorParser':
        from .nlp_color_parser import NLPColorParser
        return NLPColorParser
    elif name == 'ColorGradingParams':
        from .nlp_color_parser import ColorGradingParams
        return ColorGradingParams
    elif name == 'ColorGradingEngine':
        from .color_grading_engine import ColorGradingEngine
        return ColorGradingEngine
    elif name == 'ImageFeatureExtractor':
        from .image_retrieval import ImageFeatureExtractor
        return ImageFeatureExtractor
    elif name == 'ImageIndexDatabase':
        from .image_retrieval import ImageIndexDatabase
        return ImageIndexDatabase
    elif name == 'ImageFeature':
        from .image_retrieval import ImageFeature
        return ImageFeature
    elif name == 'AGICamera':
        from .agi_camera import AGICamera
        return AGICamera
    elif name == 'Image3DGenerator':
        from .agi_camera import Image3DGenerator
        return Image3DGenerator
    elif name == 'DepthEstimator':
        from .agi_camera import DepthEstimator
        return DepthEstimator
    elif name == 'Mesh3D':
        from .agi_camera import Mesh3D
        return Mesh3D
    elif name == 'Animation3DRenderer':
        from .agi_camera import Animation3DRenderer
        return Animation3DRenderer
    elif name == 'ObjectSegmenter':
        from .agi_camera import ObjectSegmenter
        return ObjectSegmenter
    elif name == 'StyleAnalyzer':
        from .style_analyzer import StyleAnalyzer
        return StyleAnalyzer
    elif name == 'StyleSummary':
        from .style_analyzer import StyleSummary
        return StyleSummary
    raise AttributeError(f"module 'ai' has no attribute '{name}'")


__all__ = [
    'NLPColorParser',
    'ColorGradingParams',
    'ColorGradingEngine',
    'ImageFeatureExtractor',
    'ImageIndexDatabase',
    'ImageFeature',
    'AGICamera',
    'Image3DGenerator',
    'DepthEstimator',
    'Mesh3D',
    'Animation3DRenderer',
    'ObjectSegmenter',
    'StyleAnalyzer',
    'StyleSummary',
]
