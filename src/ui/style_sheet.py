"""
应用级样式表
现代、清晰、舒适的深色科技主题。
"""


def get_dark_style() -> str:
    """获取深色主题样式。"""
    return """
    QWidget {
        background-color: #0f1113;
        color: #edf2f4;
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Segoe UI", sans-serif;
        font-size: 13px;
        selection-background-color: #18c7a7;
        selection-color: #07100f;
    }

    QMainWindow {
        background-color: #0f1113;
    }

    QMenuBar {
        background-color: #14181b;
        border-bottom: 1px solid #2b3438;
        padding: 4px 8px;
    }

    QMenuBar::item {
        background: transparent;
        border-radius: 4px;
        padding: 6px 10px;
        color: #cfd8dc;
    }

    QMenuBar::item:selected {
        background-color: #20272b;
        color: #ffffff;
    }

    QMenuBar::item:pressed {
        background-color: #18c7a7;
        color: #06110f;
    }

    QMenu {
        background-color: #181d20;
        border: 1px solid #354046;
        border-radius: 6px;
        padding: 6px;
    }

    QMenu::item {
        border-radius: 4px;
        padding: 7px 30px 7px 18px;
        color: #dfe7ea;
    }

    QMenu::item:selected {
        background-color: #203631;
        color: #7ff0d4;
    }

    QMenu::separator {
        height: 1px;
        background: #2d373c;
        margin: 6px 8px;
    }

    QToolBar {
        background-color: #14181b;
        border: none;
        border-bottom: 1px solid #2b3438;
        padding: 6px 8px;
        spacing: 6px;
    }

    QToolBar::separator {
        width: 1px;
        background: #334047;
        margin: 4px 8px;
    }

    QToolButton {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 6px 11px;
        color: #cfdbdf;
    }

    QToolButton:hover {
        background-color: #1d2428;
        border-color: #334047;
        color: #ffffff;
    }

    QToolButton:pressed,
    QToolButton:checked {
        background-color: #163d38;
        border-color: #18c7a7;
        color: #8ff7df;
    }

    QFrame#commandCenter {
        background-color: #111719;
        border-bottom: 1px solid #283438;
    }

    QLabel#commandTitle {
        color: #ffffff;
        font-size: 15px;
        font-weight: 900;
    }

    QLabel#commandSubtitle {
        color: #8fa0a7;
        font-size: 11px;
    }

    QFrame#commandDivider {
        background-color: #2c383d;
        border: none;
    }

    QLabel#commandAsset {
        color: #f3f8fa;
        font-size: 13px;
        font-weight: 800;
    }

    QLabel#commandMeta {
        color: #94a8ae;
        font-size: 11px;
    }

    QLabel#commandStatusBadge {
        background-color: #1a2226;
        border: 1px solid #334047;
        border-radius: 5px;
        color: #a7b7bd;
        font-size: 11px;
        font-weight: 800;
        padding: 5px 9px;
    }

    QLabel#commandStatusBadge[tone="active"] {
        background-color: #12342f;
        border-color: #18c7a7;
        color: #8ff7df;
    }

    QLabel#commandStatusBadge[tone="busy"] {
        background-color: #332c18;
        border-color: #a37d2d;
        color: #f0c96a;
    }

    QLabel#commandStatusBadge[tone="danger"] {
        background-color: #2a1b1d;
        border-color: #6d3a40;
        color: #ffb2b9;
    }

    QPushButton#commandButton {
        min-height: 20px;
        padding: 6px 11px;
        font-weight: 700;
    }

    QDialog#commandPalette {
        background-color: #111719;
        border: 1px solid #334047;
        border-radius: 8px;
    }

    QLabel#paletteTitle {
        color: #ffffff;
        font-size: 18px;
        font-weight: 900;
    }

    QLabel#paletteSubtitle {
        color: #92a6ac;
        font-size: 12px;
    }

    QLineEdit#paletteSearch {
        background-color: #0d1113;
        border: 1px solid #3a484f;
        border-radius: 7px;
        color: #f5fbfc;
        font-size: 14px;
        padding: 10px 12px;
    }

    QListWidget#paletteList {
        background-color: #141b1e;
        border: 1px solid #2f3d42;
        border-radius: 7px;
        padding: 6px;
    }

    QListWidget#paletteList::item {
        border-radius: 6px;
        color: #dce8eb;
        min-height: 46px;
        padding: 8px 10px;
    }

    QListWidget#paletteList::item:selected {
        background-color: #12342f;
        color: #9ff8e5;
    }

    QListWidget#paletteList::item:disabled {
        color: #5f6e74;
    }

    QLabel#paletteHint {
        color: #7d8e95;
        font-size: 11px;
    }

    QStatusBar {
        background-color: #13201f;
        border-top: 1px solid #24423e;
        color: #b9f5e8;
    }

    QStatusBar::item {
        border: none;
    }

    QGroupBox {
        background-color: #171c1f;
        border: 1px solid #2b3438;
        border-radius: 7px;
        margin-top: 15px;
        padding: 16px 12px 12px 12px;
        font-weight: 600;
        color: #edf2f4;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 7px;
        color: #7ff0d4;
        background-color: #0f1113;
    }

    QPushButton {
        background-color: #20272b;
        border: 1px solid #344047;
        border-radius: 6px;
        padding: 7px 14px;
        color: #edf2f4;
        min-height: 18px;
    }

    QPushButton:hover {
        background-color: #283136;
        border-color: #4c5d65;
    }

    QPushButton:pressed {
        background-color: #163d38;
        border-color: #18c7a7;
        color: #9ff8e5;
    }

    QPushButton:checked {
        background-color: #163d38;
        border-color: #18c7a7;
        color: #9ff8e5;
        font-weight: 600;
    }

    QPushButton:disabled {
        background-color: #161a1d;
        border-color: #252c30;
        color: #627076;
    }

    QPushButton[variant="primary"],
    QPushButton[primary="true"] {
        background-color: #18c7a7;
        border-color: #18c7a7;
        color: #041210;
        font-weight: 700;
    }

    QPushButton[variant="primary"]:hover,
    QPushButton[primary="true"]:hover {
        background-color: #35d8bc;
        border-color: #35d8bc;
    }

    QPushButton[variant="secondary"] {
        background-color: #172326;
        border-color: #315257;
        color: #baf4ea;
    }

    QPushButton[variant="danger"] {
        background-color: #2a1b1d;
        border-color: #6d3a40;
        color: #ffb2b9;
    }

    QPushButton[variant="danger"]:hover {
        background-color: #3a2025;
        border-color: #f16d7b;
        color: #ffd7db;
    }

    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox {
        background-color: #121719;
        border: 1px solid #344047;
        border-radius: 6px;
        padding: 7px 10px;
        color: #edf2f4;
    }

    QLineEdit:hover,
    QComboBox:hover,
    QSpinBox:hover,
    QDoubleSpinBox:hover {
        border-color: #4c5d65;
    }

    QLineEdit:focus,
    QComboBox:focus,
    QSpinBox:focus,
    QDoubleSpinBox:focus {
        border-color: #18c7a7;
        background-color: #151c1f;
    }

    QLineEdit:disabled,
    QComboBox:disabled,
    QSpinBox:disabled,
    QDoubleSpinBox:disabled {
        background-color: #15191b;
        border-color: #252c30;
        color: #657278;
    }

    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border: none;
        background: transparent;
    }

    QComboBox::down-arrow {
        width: 0;
        height: 0;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #9fb0b7;
    }

    QComboBox QAbstractItemView {
        background-color: #171c1f;
        border: 1px solid #344047;
        border-radius: 6px;
        selection-background-color: #163d38;
        selection-color: #9ff8e5;
        outline: none;
    }

    QComboBox QAbstractItemView::item {
        min-height: 28px;
        padding: 6px 10px;
    }

    QSlider::groove:horizontal {
        height: 5px;
        background: #293236;
        border-radius: 2px;
    }

    QSlider::sub-page:horizontal {
        background: #18c7a7;
        border-radius: 2px;
    }

    QSlider::handle:horizontal {
        width: 16px;
        height: 16px;
        margin: -6px 0;
        background: #f5fbfc;
        border: 2px solid #18c7a7;
        border-radius: 8px;
    }

    QSlider::handle:horizontal:hover {
        border-color: #72d7ff;
    }

    QScrollArea,
    QAbstractScrollArea {
        background-color: #0f1113;
        border: none;
    }

    QScrollBar:vertical {
        background: #121719;
        width: 11px;
        border: none;
        margin: 0;
    }

    QScrollBar::handle:vertical {
        background: #3c4a50;
        border-radius: 5px;
        min-height: 32px;
        margin: 2px;
    }

    QScrollBar::handle:vertical:hover {
        background: #53646b;
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0;
    }

    QScrollBar:horizontal {
        background: #121719;
        height: 11px;
        border: none;
        margin: 0;
    }

    QScrollBar::handle:horizontal {
        background: #3c4a50;
        border-radius: 5px;
        min-width: 32px;
        margin: 2px;
    }

    QScrollBar::handle:horizontal:hover {
        background: #53646b;
    }

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        width: 0;
    }

    QTabWidget::pane {
        border: none;
        background-color: #101416;
    }

    QTabBar::tab {
        background-color: #14191c;
        border: 1px solid #2b3438;
        border-bottom: 1px solid #2b3438;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 9px 14px;
        margin-right: 4px;
        color: #9fb0b7;
    }

    QTabBar::tab:selected {
        background-color: #1b2225;
        border-color: #355c5a;
        color: #ffffff;
        border-bottom: 2px solid #18c7a7;
    }

    QTabBar::tab:hover:!selected {
        background-color: #1b2225;
        color: #dce8eb;
    }

    QProgressBar {
        background-color: #20272b;
        border: none;
        border-radius: 4px;
        height: 7px;
        text-align: center;
        color: transparent;
    }

    QProgressBar::chunk {
        background-color: #18c7a7;
        border-radius: 4px;
    }

    QSplitter::handle {
        background-color: #1f282c;
    }

    QSplitter::handle:horizontal {
        width: 4px;
    }

    QSplitter::handle:vertical {
        height: 4px;
    }

    QSplitter::handle:hover {
        background-color: #18c7a7;
    }

    QLabel {
        background: transparent;
    }

    QLabel#sectionHint,
    QLabel#mutedText {
        color: #8fa0a7;
        font-size: 12px;
    }

    QLabel#dialogTitle {
        color: #f3f8fa;
        font-size: 14px;
        font-weight: 800;
        padding: 8px 0;
    }

    QLabel#selectionSummary {
        background-color: #121719;
        border: 1px solid #2b3438;
        border-radius: 7px;
        color: #d9e4e8;
        padding: 10px;
        min-height: 42px;
    }

    QLabel#selectionStatus[state="idle"] {
        color: #8fa0a7;
        font-size: 12px;
    }

    QLabel#selectionStatus[state="success"] {
        color: #74e3c9;
        font-size: 12px;
        font-weight: 700;
    }

    QLabel#selectionStatus[state="error"] {
        color: #ff8792;
        font-size: 12px;
        font-weight: 700;
    }

    QLabel#inlineStatus[state="muted"] {
        color: #8fa0a7;
        font-size: 11px;
    }

    QLabel#inlineStatus[state="success"] {
        color: #74e3c9;
        font-size: 11px;
        font-weight: 700;
    }

    QLabel#inlineStatus[state="error"] {
        color: #ff8792;
        font-size: 11px;
        font-weight: 700;
    }

    QWidget#imageViewport,
    QWidget#selectionViewport,
    QLabel#animationPreview {
        background-color: #090b0c;
        border: 1px solid #283236;
        border-radius: 7px;
    }

    QFrame#imageCanvas {
        background-color: #090b0c;
        border: 1px solid #283236;
        border-radius: 7px;
    }

    QWidget#viewerOverlay {
        background: transparent;
    }

    QFrame#canvasHud {
        background-color: rgba(16, 22, 24, 218);
        border: 1px solid #2f3d42;
        border-radius: 7px;
    }

    QLabel#canvasAssetName {
        color: #f5fbfc;
        font-size: 13px;
        font-weight: 800;
    }

    QLabel#canvasAssetMeta {
        color: #92a6ac;
        font-size: 11px;
    }

    QLabel#canvasBadge {
        background-color: #1a2226;
        border: 1px solid #334047;
        border-radius: 5px;
        color: #a7b7bd;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 8px;
    }

    QLabel#canvasBadge[tone="active"] {
        background-color: #12342f;
        border-color: #18c7a7;
        color: #8ff7df;
    }

    QLabel#canvasBadge[tone="busy"] {
        background-color: #332c18;
        border-color: #a37d2d;
        color: #f0c96a;
    }

    QLabel#canvasBadge[tone="danger"] {
        background-color: #2a1b1d;
        border-color: #6d3a40;
        color: #ffb2b9;
    }

    QFrame#viewerEmptyState {
        background-color: rgba(20, 27, 30, 232);
        border: 1px solid #304047;
        border-radius: 8px;
        min-width: 390px;
        max-width: 520px;
    }

    QLabel#viewerEmptyTitle {
        color: #ffffff;
        font-size: 18px;
        font-weight: 800;
    }

    QLabel#viewerEmptySubtitle {
        color: #a8b8be;
        font-size: 12px;
        line-height: 150%;
    }

    QLabel#viewerDropHint {
        color: #74e3c9;
        font-size: 11px;
    }

    QFrame#viewerProcessingState {
        background-color: rgba(16, 22, 24, 238);
        border: 1px solid #3a5959;
        border-radius: 8px;
        min-width: 330px;
        max-width: 480px;
    }

    QLabel#viewerProcessingTitle {
        color: #ffffff;
        font-size: 17px;
        font-weight: 900;
    }

    QLabel#viewerProcessingDetail {
        color: #b8c7cc;
        font-size: 12px;
    }

    QLabel#viewerProcessingHint {
        color: #74e3c9;
        font-size: 11px;
        font-weight: 700;
    }

    QFrame#libraryHeader {
        background-color: #141a1d;
        border: 1px solid #2b3539;
        border-radius: 8px;
    }

    QLabel#libraryTitle {
        color: #f5fbfc;
        font-size: 18px;
        font-weight: 900;
    }

    QLabel#librarySubtitle {
        color: #9eb0b7;
        font-size: 12px;
    }

    QLabel#libraryStateBadge {
        background-color: #24272a;
        border: 1px solid #3a4248;
        border-radius: 5px;
        color: #c5cdd1;
        font-size: 11px;
        font-weight: 800;
        padding: 4px 8px;
    }

    QLabel#libraryStateBadge[tone="ready"] {
        background-color: #12342f;
        border-color: #18c7a7;
        color: #8ff7df;
    }

    QLabel#libraryStateBadge[tone="offline"] {
        background-color: #2a1b1d;
        border-color: #6d3a40;
        color: #ffb2b9;
    }

    QFrame#librarySummaryStrip {
        background-color: #101517;
        border: 1px solid #253035;
        border-radius: 7px;
    }

    QFrame#libraryMetric {
        background-color: transparent;
        border: none;
    }

    QLabel#libraryMetricName {
        color: #809199;
        font-size: 11px;
    }

    QLabel#libraryMetricValue {
        color: #f1f7f9;
        font-size: 13px;
        font-weight: 800;
    }

    QLabel#librarySummaryText {
        color: #9eb0b7;
        font-size: 12px;
    }

    QFrame#libraryEmptyState {
        background-color: #121719;
        border: 1px solid #2c383d;
        border-radius: 8px;
        min-height: 180px;
    }

    QLabel#libraryEmptyTitle {
        color: #f4fafb;
        font-size: 16px;
        font-weight: 900;
    }

    QLabel#libraryEmptySubtitle {
        color: #9eb0b7;
        font-size: 12px;
    }

    QDialog#libraryManagerDialog {
        background-color: #0f1113;
    }

    QFrame#libraryManagerHeader {
        background-color: #141a1d;
        border: 1px solid #2b3539;
        border-radius: 8px;
    }

    QLabel#libraryManagerTitle {
        color: #f5fbfc;
        font-size: 19px;
        font-weight: 900;
    }

    QLabel#libraryManagerSubtitle {
        color: #9eb0b7;
        font-size: 12px;
    }

    QFrame#libraryManagerMetric {
        background-color: #101517;
        border: 1px solid #253035;
        border-radius: 7px;
        min-width: 72px;
        padding: 7px 9px;
    }

    QLabel#libraryManagerMetricName {
        color: #809199;
        font-size: 11px;
    }

    QLabel#libraryManagerMetricValue {
        color: #f1f7f9;
        font-size: 13px;
        font-weight: 800;
    }

    QWidget#libraryManagerListPane {
        background-color: transparent;
    }

    QListWidget#libraryManagerList {
        background-color: #111719;
        border: 1px solid #2b3539;
        border-radius: 8px;
        padding: 8px;
    }

    QListWidget#libraryManagerList::item {
        border-radius: 6px;
        color: #d9e4e8;
        padding: 6px;
    }

    QListWidget#libraryManagerList::item:selected {
        background-color: #12342f;
        color: #9ff8e5;
    }

    QGroupBox#libraryManagerDetailPanel {
        background-color: #121719;
        border: 1px solid #2b3539;
        border-radius: 8px;
        margin-top: 10px;
        padding: 12px;
        color: #dce7eb;
        font-weight: 800;
    }

    QGroupBox#libraryManagerDetailPanel::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 6px;
    }

    QLabel#libraryManagerPreview {
        background-color: #090b0c;
        border: 1px solid #283236;
        border-radius: 7px;
        color: #7e9199;
        font-weight: 700;
    }

    QDialog#imagePickerDialog {
        background-color: #0f1113;
    }

    QFrame#imagePickerHeader {
        background-color: #141a1d;
        border: 1px solid #2b3539;
        border-radius: 8px;
    }

    QLabel#imagePickerTitle {
        color: #f5fbfc;
        font-size: 19px;
        font-weight: 900;
    }

    QLabel#imagePickerSubtitle {
        color: #9eb0b7;
        font-size: 12px;
    }

    QFrame#imagePickerMetric {
        background-color: #101517;
        border: 1px solid #253035;
        border-radius: 7px;
        min-width: 74px;
        padding: 7px 9px;
    }

    QLabel#imagePickerMetricName {
        color: #809199;
        font-size: 11px;
    }

    QLabel#imagePickerMetricValue {
        color: #f1f7f9;
        font-size: 13px;
        font-weight: 800;
    }

    QTabWidget#imagePickerTabs::pane {
        background-color: #101416;
        border: 1px solid #273236;
        border-radius: 8px;
        top: -1px;
    }

    QFrame#viewerToolbar {
        background-color: #14191c;
        border-top: 1px solid #2b3438;
    }

    QLabel#zoomLabel {
        color: #b7c4c9;
        font-weight: 600;
    }

    QLabel#thumbnailImage {
        background-color: #111619;
        border: 2px solid #2c373c;
        border-radius: 7px;
    }

    QLabel#thumbnailName {
        color: #b8c5ca;
        font-size: 11px;
    }

    QLabel#thumbnailImage[selected="true"] {
        border-color: #18c7a7;
        background-color: #162321;
    }

    QLabel#thumbnailName[selected="true"] {
        color: #8ff7df;
        font-weight: 700;
    }

    QWidget#workflowPanel {
        background-color: #101416;
    }

    QLabel#workflowTitle {
        color: #ffffff;
        font-size: 19px;
        font-weight: 800;
    }

    QLabel#workflowSubtitle {
        color: #9fb0b7;
        font-size: 12px;
    }

    QLabel#workflowMetricName {
        color: #8fa0a7;
        font-size: 12px;
    }

    QLabel#workflowMetricValue {
        color: #f3f8fa;
        font-weight: 700;
    }

    QLabel#workflowStep {
        color: #9fb0b7;
        padding: 2px 0;
    }

    QLabel#workflowStep[complete="true"] {
        color: #74e3c9;
        font-weight: 700;
    }

    QFrame#workflowModelRow {
        background-color: transparent;
        border: none;
    }

    QLabel#workflowModelState {
        color: #9fb0b7;
        font-size: 12px;
    }

    QLabel#workflowModelState[state="加载中"] {
        color: #f0c96a;
    }

    QLabel#workflowModelState[state="就绪"] {
        color: #74e3c9;
        font-weight: 700;
    }

    QLabel#workflowModelState[state="失败"] {
        color: #ff8792;
        font-weight: 700;
    }

    QListWidget {
        background-color: #121719;
        border: 1px solid #2b3438;
        border-radius: 6px;
        padding: 4px;
        outline: none;
    }

    QListWidget::item {
        color: #d9e4e8;
        border-radius: 4px;
        padding: 5px 4px;
    }

    QListWidget::item:selected {
        background-color: #163d38;
        color: #9ff8e5;
    }

    QFrame {
        border: none;
    }

    QMessageBox {
        background-color: #171c1f;
    }

    QMessageBox QLabel {
        color: #edf2f4;
    }

    QToolTip {
        background-color: #172326;
        border: 1px solid #315257;
        border-radius: 4px;
        color: #edf2f4;
        padding: 6px;
    }
    """


def get_light_style() -> str:
    """浅色主题占位，保留接口稳定。"""
    return """
    QWidget {
        background-color: #f7fafb;
        color: #172023;
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "Segoe UI", sans-serif;
        font-size: 13px;
    }
    """
