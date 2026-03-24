"""
UI工具类
"""
from PySide6.QtCore import QObject, QEvent

class WheelBlocker(QObject):
    """屏蔽鼠标滚轮事件的过滤器"""
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            return True
        return False
