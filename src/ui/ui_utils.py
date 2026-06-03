"""
UI工具类
"""
from PySide6.QtCore import QObject, QEvent


def fit_thumbnail_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Return a proportional thumbnail size with neither dimension below 1px."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = max_size / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


class WheelBlocker(QObject):
    """屏蔽鼠标滚轮事件的过滤器"""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            return True
        return False
