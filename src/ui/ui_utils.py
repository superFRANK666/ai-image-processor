"""
UI工具类
"""
from PySide6.QtCore import QObject, QEvent
from PySide6.QtWidgets import QApplication


def fit_thumbnail_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    """Return a proportional thumbnail size with neither dimension below 1px."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = max_size / max(width, height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def fit_within_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    """Return a proportional size within a bounding box, keeping dimensions valid."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("Target dimensions must be positive")

    scale = min(max_width / width, max_height / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


class WheelBlocker(QObject):
    """屏蔽控件自身的鼠标滚轮事件，并将滚轮事件转发给父级控件（如滚动区域），
    从而让页面可以正常通过滚轮滚动，但被安装该过滤器的控件（滑块、下拉框等）
    不会因滚轮而改变其自身数值。"""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            # 将事件转发给父级控件，让滚动区域可以正常响应滚轮
            parent = obj.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, event)
            return True  # 阻止 obj 自身处理滚轮
        return False
