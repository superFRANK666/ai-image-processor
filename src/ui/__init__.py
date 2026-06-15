"""
UI模块初始化
使用延迟导入优化启动速度 — 所有成员均按需导入
"""

__all__ = [
    'main',
    'MainWindow',
    'ImageViewer',
    'ColorGradingPanel',
    'AGICameraPanel',
    'ImageLibraryPanel',
    'get_dark_style',
    'get_light_style',
]


def __getattr__(name):
    """延迟导入机制 — 任何成员均在首次访问时才导入"""
    if name in ('main', 'MainWindow'):
        from .main_window import MainWindow, main
        # 缓存到模块命名空间，避免重复触发 __getattr__
        import src.ui as _mod
        _mod.MainWindow = MainWindow
        _mod.main = main
        return MainWindow if name == 'MainWindow' else main
    elif name == 'ImageViewer':
        from .image_viewer import ImageViewer
        return ImageViewer
    elif name == 'ColorGradingPanel':
        from .color_grading_panel import ColorGradingPanel
        return ColorGradingPanel
    elif name == 'AGICameraPanel':
        from .agi_camera_panel import AGICameraPanel
        return AGICameraPanel
    elif name == 'ImageLibraryPanel':
        from .image_library_panel import ImageLibraryPanel
        return ImageLibraryPanel
    elif name == 'get_dark_style':
        from .style_sheet import get_dark_style
        return get_dark_style
    elif name == 'get_light_style':
        from .style_sheet import get_light_style
        return get_light_style
    raise AttributeError(f"模块 'ui' 中没有属性 '{name}'")
