"""
UI 字体选择工具。
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtGui import QFont, QFontDatabase


PREFERRED_UI_FONTS = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "Segoe UI",
    "Arial",
)


def choose_ui_font(families: Optional[Iterable[str]] = None) -> str:
    """从系统可用字体中选择适合中文界面的字体。"""
    available = set(families if families is not None else QFontDatabase.families())
    for family in PREFERRED_UI_FONTS:
        if family in available:
            return family
    return "Microsoft YaHei"


def apply_application_font(app, point_size: int = 10) -> QFont:
    """为 QApplication 设置默认 UI 字体并返回该字体。"""
    font = QFont()
    font.setFamily(choose_ui_font())
    font.setPointSize(point_size)
    app.setFont(font)
    return font
