#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 全模态影像处理软件主入口。
"""
import argparse
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


def check_dependencies(full_check: bool = False) -> bool:
    """
    检查运行依赖。

    Args:
        full_check: 是否额外检查可选/高级依赖。

    Returns:
        核心依赖是否齐全。
    """
    missing = []

    # 核心依赖（应用启动必需）。
    core_deps = {
        "PySide6": "PySide6",
        "numpy": "numpy",
        "cv2": "opencv-python",
    }

    # 可选依赖（高级功能使用）。
    optional_deps = {
        "PIL": "Pillow",
        "torch": "torch",
        "transformers": "transformers",
        "sentence_transformers": "sentence-transformers",
        "chromadb": "chromadb",
        "onnxruntime": "onnxruntime",
        "open3d": "open3d",
    }

    logger.info("正在检查核心依赖...")
    for import_name, package_name in core_deps.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        logger.error("缺少核心依赖:")
        for dep in missing:
            print(f"  - {dep}")
        print("\n请运行: pip install -r requirements.txt")
        return False

    if full_check:
        optional_missing = []
        logger.info("正在检查可选/高级依赖...")
        for import_name, package_name in optional_deps.items():
            try:
                __import__(import_name)
            except ImportError:
                optional_missing.append(package_name)

        if optional_missing:
            print("以下可选依赖未安装（部分高级功能可能不可用）:")
            for dep in optional_missing:
                print(f"  - {dep}")
        else:
            print("所有可选依赖检查通过。")

    return True


def main() -> None:
    """应用主函数。"""
    parser = argparse.ArgumentParser(description="AI全模态影像处理软件")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--check-deps", action="store_true", help="仅检查依赖并退出")
    args = parser.parse_args()

    try:
        from src.core.config import APP_VERSION
    except Exception as exc:
        logger.warning(f"无法读取版本配置: {exc}")
        APP_VERSION = "1.0.0"

    print("========================================")
    print(f"   AI 影像处理软件 v{APP_VERSION}")
    print("========================================")

    if args.check_deps:
        if check_dependencies(full_check=True):
            print("依赖检查完成。")
            sys.exit(0)
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    if args.debug:
        os.environ["QT_LOGGING_RULES"] = "*.debug=true"
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    else:
        os.environ["QT_LOGGING_RULES"] = "*.debug=false"
        logger.setLevel(logging.INFO)

    logger.info("正在加载用户界面...")
    try:
        from src.ui import main as ui_main

        ui_main()
    except Exception as exc:
        logger.critical(f"程序启动失败: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
