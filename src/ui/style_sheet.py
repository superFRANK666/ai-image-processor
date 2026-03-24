"""
样式表
深色主题样式
"""


def get_dark_style() -> str:
    """获取深色主题样式"""
    return """
    /* 全局样式 */
    QWidget {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 13px;
    }

    /* 主窗口 */
    QMainWindow {
        background-color: #1e1e1e;
    }

    /* 菜单栏 */
    QMenuBar {
        background-color: #2d2d2d;
        border-bottom: 1px solid #404040;
        padding: 2px;
    }

    QMenuBar::item {
        padding: 5px 10px;
        background: transparent;
    }

    QMenuBar::item:selected {
        background-color: #404040;
    }

    QMenuBar::item:pressed {
        background-color: #0078d4;
    }

    /* 菜单 */
    QMenu {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        padding: 5px;
    }

    QMenu::item {
        padding: 5px 30px 5px 20px;
    }

    QMenu::item:selected {
        background-color: #0078d4;
    }

    QMenu::separator {
        height: 1px;
        background: #404040;
        margin: 5px 10px;
    }

    /* 工具栏 */
    QToolBar {
        background-color: #2d2d2d;
        border: none;
        border-bottom: 1px solid #404040;
        padding: 3px;
        spacing: 5px;
    }

    QToolBar::separator {
        width: 1px;
        background: #404040;
        margin: 5px;
    }

    QToolButton {
        background: transparent;
        border: none;
        border-radius: 3px;
        padding: 5px 10px;
        color: #e0e0e0;
    }

    QToolButton:hover {
        background-color: #404040;
    }

    QToolButton:pressed {
        background-color: #0078d4;
    }

    QToolButton:checked {
        background-color: #0078d4;
    }

    /* 状态栏 */
    QStatusBar {
        background-color: #007acc;
        color: white;
        border: none;
    }

    QStatusBar::item {
        border: none;
    }

    /* 分组框 */
    QGroupBox {
        border: 1px solid #404040;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 5px;
        color: #0078d4;
    }

    /* 按钮 */
    QPushButton {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 15px;
        color: #e0e0e0;
    }

    QPushButton:hover {
        background-color: #4a4a4a;
        border-color: #666666;
    }

    QPushButton:pressed {
        background-color: #0078d4;
        border-color: #0078d4;
    }

    QPushButton:disabled {
        background-color: #2d2d2d;
        color: #666666;
    }

    /* 主要按钮 */
    QPushButton[primary="true"] {
        background-color: #0078d4;
        border-color: #0078d4;
    }

    QPushButton[primary="true"]:hover {
        background-color: #1a86d9;
    }

    /* 输入框 */
    QLineEdit {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 6px 10px;
        color: #e0e0e0;
        selection-background-color: #0078d4;
    }

    QLineEdit:focus {
        border-color: #0078d4;
    }

    QLineEdit:disabled {
        background-color: #2d2d2d;
        color: #666666;
    }

    /* 滑块 */
    QSlider::groove:horizontal {
        height: 4px;
        background: #555555;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        width: 14px;
        height: 14px;
        margin: -5px 0;
        background: #0078d4;
        border-radius: 7px;
    }

    QSlider::handle:horizontal:hover {
        background: #1a86d9;
    }

    QSlider::sub-page:horizontal {
        background: #0078d4;
        border-radius: 2px;
    }

    /* 下拉框 */
    QComboBox {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px 10px;
        padding-right: 25px;
        color: #e0e0e0;
    }

    QComboBox:hover {
        border-color: #666666;
    }

    QComboBox:focus {
        border-color: #0078d4;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border: none;
        background: transparent;
    }

    QComboBox::down-arrow {
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #e0e0e0;
    }

    QComboBox::down-arrow:hover {
        border-top-color: #ffffff;
    }

    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        selection-background-color: #0078d4;
        outline: none;
    }

    QComboBox QAbstractItemView::item {
        padding: 5px 10px;
        min-height: 25px;
    }

    QComboBox QAbstractItemView::item:hover {
        background-color: #404040;
    }

    QComboBox QAbstractItemView::item:selected {
        background-color: #0078d4;
    }

    /* 数值输入框 */
    QSpinBox, QDoubleSpinBox {
        background-color: #3c3c3c;
        border: 1px solid #555555;
        border-radius: 4px;
        padding: 5px;
        color: #e0e0e0;
    }

    QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #0078d4;
    }

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        background: #4a4a4a;
        border: none;
        width: 16px;
    }

    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background: #555555;
    }

    /* 滚动区域 */
    QScrollArea {
        border: none;
        background-color: #1e1e1e;
    }

    /* 滚动条 */
    QScrollBar:vertical {
        background: #2d2d2d;
        width: 12px;
        border: none;
    }

    QScrollBar::handle:vertical {
        background: #555555;
        border-radius: 6px;
        min-height: 30px;
        margin: 2px;
    }

    QScrollBar::handle:vertical:hover {
        background: #666666;
    }

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
    }

    QScrollBar:horizontal {
        background: #2d2d2d;
        height: 12px;
        border: none;
    }

    QScrollBar::handle:horizontal {
        background: #555555;
        border-radius: 6px;
        min-width: 30px;
        margin: 2px;
    }

    QScrollBar::handle:horizontal:hover {
        background: #666666;
    }

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0;
    }

    /* 标签页 */
    QTabWidget::pane {
        border: 1px solid #404040;
        border-top: none;
        background-color: #1e1e1e;
    }

    QTabBar::tab {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-bottom: none;
        padding: 8px 15px;
        margin-right: 2px;
    }

    QTabBar::tab:selected {
        background-color: #1e1e1e;
        border-bottom: 2px solid #0078d4;
    }

    QTabBar::tab:hover:!selected {
        background-color: #3c3c3c;
    }

    /* 停靠窗口 */
    QDockWidget {
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }

    QDockWidget::title {
        background-color: #2d2d2d;
        padding: 8px;
        border-bottom: 1px solid #404040;
    }

    QDockWidget::close-button, QDockWidget::float-button {
        background: transparent;
        border: none;
        padding: 2px;
    }

    QDockWidget::close-button:hover, QDockWidget::float-button:hover {
        background: #404040;
    }

    /* 进度条 */
    QProgressBar {
        background-color: #3c3c3c;
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
    }

    QProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 3px;
    }

    /* 分割器 */
    QSplitter::handle {
        background-color: #404040;
    }

    QSplitter::handle:horizontal {
        width: 3px;
    }

    QSplitter::handle:vertical {
        height: 3px;
    }

    QSplitter::handle:hover {
        background-color: #0078d4;
    }

    /* 标签 */
    QLabel {
        background: transparent;
    }

    /* 框架 */
    QFrame {
        border: none;
    }

    /* 消息框 */
    QMessageBox {
        background-color: #2d2d2d;
    }

    QMessageBox QLabel {
        color: #e0e0e0;
    }

    /* 工具提示 */
    QToolTip {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        color: #e0e0e0;
        padding: 5px;
    }
    """


def get_light_style() -> str:
    """获取浅色主题样式"""
    return """
    QWidget {
        background-color: #f5f5f5;
        color: #333333;
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 13px;
    }

    /* 更多浅色主题样式... */
    """
