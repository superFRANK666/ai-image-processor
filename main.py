#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 全模态影像处理软件主入口。

启动优化策略：
1. 先创建 QApplication + Splash Screen，让用户立刻看到界面
2. 在 Splash 存在期间完成所有重量级 import
3. 主窗口就绪后关闭 Splash
"""
import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()

# 确保当前工作目录是项目根目录，保证相对路径与资源路径稳定。
if os.getcwd() != str(PROJECT_ROOT):
    os.chdir(str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Main")


def check_dependencies() -> bool:
    """
    检查核心运行依赖。
    使用 find_spec 探测包是否存在，避免为了检查而提前导入 numpy/cv2。
    """
    missing = []
    core_deps = {
        "PySide6": "PySide6",
        "numpy": "numpy",
        "cv2": "opencv-python",
    }
    for import_name, package_name in core_deps.items():
        if import_name not in sys.modules and importlib.util.find_spec(import_name) is None:
            missing.append(package_name)

    if missing:
        logger.error("缺少核心依赖: %s", ", ".join(missing))
        print("\n请运行: pip install -r requirements.txt")
        return False
    return True


def _show_splash(app):
    """创建并显示 Splash Screen，返回 splash 对象。"""
    try:
        from PySide6.QtWidgets import QSplashScreen
        from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient
        from PySide6.QtCore import Qt, QRect

        # 尝试从文件加载 splash 图像
        splash_path = PROJECT_ROOT / "resources" / "splash.png"

        if splash_path.exists():
            pixmap = QPixmap(str(splash_path))
        else:
            # 动态生成高质量渐变 splash 画面
            pixmap = QPixmap(600, 340)
            pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 背景渐变
            gradient = QLinearGradient(0, 0, 600, 340)
            gradient.setColorAt(0.0, QColor(15, 17, 26))
            gradient.setColorAt(0.5, QColor(20, 24, 40))
            gradient.setColorAt(1.0, QColor(12, 14, 22))
            painter.fillRect(QRect(0, 0, 600, 340), gradient)

            # 顶部装饰线
            accent = QLinearGradient(0, 0, 600, 0)
            accent.setColorAt(0.0, QColor(80, 120, 255, 0))
            accent.setColorAt(0.3, QColor(100, 140, 255, 220))
            accent.setColorAt(0.7, QColor(180, 100, 255, 220))
            accent.setColorAt(1.0, QColor(180, 100, 255, 0))
            from PySide6.QtGui import QPen, QBrush
            painter.setPen(QPen(QBrush(accent), 2))
            painter.drawLine(0, 3, 600, 3)

            # 标题
            font = QFont()
            font.setFamily("Microsoft YaHei UI")
            font.setPixelSize(32)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(240, 240, 255))
            painter.drawText(QRect(0, 100, 600, 50), Qt.AlignmentFlag.AlignHCenter, "AI 影像处理")

            # 副标题
            font.setPixelSize(14)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(130, 145, 180))
            painter.drawText(QRect(0, 155, 600, 30), Qt.AlignmentFlag.AlignHCenter, "全模态 · 端侧 AI · 智能调色")

            # 版本号
            font.setPixelSize(12)
            painter.setFont(font)
            painter.setPen(QColor(80, 90, 120))
            try:
                from src.core.config import APP_VERSION
            except Exception:
                APP_VERSION = ""
            if APP_VERSION:
                painter.drawText(QRect(0, 305, 600, 25), Qt.AlignmentFlag.AlignHCenter, f"v{APP_VERSION}")

            # 加载提示
            painter.setPen(QColor(100, 120, 180))
            font.setPixelSize(11)
            painter.setFont(font)
            painter.drawText(QRect(0, 270, 600, 25), Qt.AlignmentFlag.AlignHCenter, "正在启动，请稍候…")

            painter.end()

        splash = QSplashScreen(pixmap)
        splash.setWindowFlags(
            splash.windowFlags() |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint
        )
        splash.show()
        app.processEvents()
        return splash
    except Exception as exc:
        logger.warning("无法创建启动画面: %s", exc)
        return None


def main() -> None:
    """应用主函数。"""
    parser = argparse.ArgumentParser(description="AI全模态影像处理软件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--check-deps", action="store_true", help="仅检查依赖并退出")
    args = parser.parse_args()

    # ── 第一步：创建 QApplication（必须在任何 Qt 对象之前）────────────────
    # 这一步很快，为 Splash Screen 做准备
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AI影像处理")

    if args.check_deps:
        if check_dependencies():
            print("依赖检查完成。")
            sys.exit(0)
        sys.exit(1)

    # ── 第二步：显示 Splash Screen，让用户立刻感知程序已启动 ─────────────
    splash = _show_splash(app)

    # ── 第三步：在 Splash 显示期间做所有重量级初始化 ─────────────────────
    if not check_dependencies():
        if splash:
            splash.close()
        sys.exit(1)

    try:
        from src.core.config import APP_VERSION
    except Exception as exc:
        logger.warning("无法读取版本配置: %s", exc)
        APP_VERSION = "1.2.0"

    if splash:
        splash.showMessage(
            f"  AI 影像处理 v{APP_VERSION}  正在加载界面...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(100, 120, 180)
        )
        app.processEvents()

    if args.debug:
        os.environ["QT_LOGGING_RULES"] = "*.debug=true"
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    else:
        os.environ["QT_LOGGING_RULES"] = "*.debug=false"
        logger.setLevel(logging.INFO)

    logger.info("正在加载用户界面...")
    try:
        from src.ui.main_window import MainWindow

        window = MainWindow()
        window.showMaximized()

        # 主窗口就绪后关闭 Splash
        if splash:
            splash.finish(window)

        sys.exit(app.exec())
    except Exception as exc:
        logger.critical("程序启动失败: %s", exc, exc_info=True)
        if splash:
            splash.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
