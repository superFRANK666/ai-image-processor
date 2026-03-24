"""
主窗口
AI全模态影像处理软件主界面
"""
import sys
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe, imwrite as imwrite_safe

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QProgressBar, QLabel, QApplication, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QAction, QKeySequence, QFont

from .image_viewer import ImageViewer
from .color_grading_panel import ColorGradingPanel
from .agi_camera_panel import AGICameraPanel
from .image_library_panel import ImageLibraryPanel
from .library_manager_dialog import LibraryManagerDialog
from .image_picker_dialog import pick_images
from .style_sheet import get_dark_style

# 导入配置 (使用相对导入)
from ..core.config import UI_CONFIG, SUPPORTED_FORMATS, IMAGE_INDEX_DIR, APP_VERSION
from ..core.model_manager import ModelManager

# 延迟导入 AI 模块（类型检查时导入,运行时延迟）
if TYPE_CHECKING:
    from ..ai import ColorGradingEngine, NLPColorParser, AGICamera, ImageIndexDatabase, StyleAnalyzer


class ProcessingThread(QThread):
    """后台处理线程"""
    finished = Signal(object)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_running = False
        self._stop_requested = False

    def run(self):
        self._is_running = True
        try:
            result = self.func(*self.args, **self.kwargs)
            if not self._stop_requested:
                self.finished.emit(result)
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            # 捕获常见的应用程序错误,不捕获系统信号
            if not self._stop_requested:
                self.error.emit(str(e))
        finally:
            self._is_running = False

    def is_running(self) -> bool:
        return self._is_running
    
    def request_stop(self):
        """请求线程停止"""
        self._stop_requested = True


class MainWindow(QMainWindow):
    """主窗口"""

    # 历史记录最大数量
    MAX_HISTORY = 20

    def __init__(self):
        super().__init__()

        # 初始化组件
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.current_file_path: Optional[str] = None
        self.thread: Optional[ProcessingThread] = None

        # 历史记录栈（用于撤销功能）
        # 每个记录包含: {'image': np.ndarray, 'params': ColorGradingParams}
        self._history_stack = []
        self._current_params = None  # 当前调色参数

        # 使用 ModelManager 统一管理 AI 模型
        self.model_manager = ModelManager(self)
        self._register_models()

        # AI引擎引用（通过 ModelManager 获取）
        self.color_engine: Optional[ColorGradingEngine] = None
        self.nlp_parser: Optional[NLPColorParser] = None
        self.agi_camera: Optional[AGICamera] = None
        self.image_db: Optional[ImageIndexDatabase] = None
        self.style_analyzer: Optional['StyleAnalyzer'] = None

        # 连接 ModelManager 信号
        self.model_manager.model_loaded.connect(self._on_model_loaded)
        self.model_manager.model_failed.connect(self._on_model_failed)
        self.model_manager.loading_started.connect(self._on_loading_started)

        # 设置UI
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._connect_signals()

        # 应用样式
        self.setStyleSheet(get_dark_style())

        # 延迟加载图像数据库（在窗口显示后加载，避免阻塞启动）
        # 从100ms增加到1000ms，让窗口先显示，用户体验更好
        QTimer.singleShot(1000, self._deferred_init)

    def _register_models(self):
        """注册所有 AI 模型到 ModelManager"""
        # 注册调色引擎
        self.model_manager.register_model('color_engine', self._create_color_engine)

        # 注册 NLP 解析器
        self.model_manager.register_model('nlp_parser', self._create_nlp_parser)

        # 注册 AGI 相机
        self.model_manager.register_model('agi_camera', self._create_agi_camera)

        # 注册图像数据库
        self.model_manager.register_model('image_db', self._create_image_db)

        # 注册风格分析器
        self.model_manager.register_model('style_analyzer', self._create_style_analyzer)

    def _create_color_engine(self):
        """创建调色引擎实例"""
        from ..ai import ColorGradingEngine
        return ColorGradingEngine()

    def _create_nlp_parser(self):
        """创建 NLP 解析器实例"""
        from ..core.config_loader import load_llm_config
        from ..ai import NLPColorParser

        llm_config = load_llm_config()

        if llm_config.get("enabled", False):
            # 使用LLM模式
            use_llm = True
            llm_params = {k: v for k, v in llm_config.items() if k != "enabled"}
            return NLPColorParser(use_llm=use_llm, llm_config=llm_params)
        else:
            # 传统模式
            return NLPColorParser(use_llm=False)

    def _create_agi_camera(self):
        """创建 AGI 相机实例"""
        from ..ai import AGICamera
        return AGICamera()

    def _create_image_db(self):
        """创建图像数据库实例"""
        from ..ai import ImageIndexDatabase
        return ImageIndexDatabase(IMAGE_INDEX_DIR)

    def _create_style_analyzer(self):
        """创建风格分析器实例"""
        from ..ai import StyleAnalyzer
        # 确保 image_db 已加载
        if not self.image_db:
            self.image_db = self.model_manager.get_model('image_db')
            if not self.image_db:
                self.model_manager.load_model_sync('image_db')
                self.image_db = self.model_manager.get_model('image_db')
        return StyleAnalyzer(self.image_db.feature_extractor)

    def _on_loading_started(self, model_name: str):
        """模型开始加载时的回调"""
        display_names = {
            'color_engine': '调色引擎',
            'nlp_parser': '语义理解模型',
            'agi_camera': '3D处理模块',
            'image_db': '图像数据库',
            'style_analyzer': '风格分析模型'
        }
        display_name = display_names.get(model_name, model_name)
        self.statusbar.showMessage(f"正在加载{display_name}...")

    def _on_model_loaded(self, model_name: str):
        """模型加载完成时的回调"""
        display_names = {
            'color_engine': '调色引擎',
            'nlp_parser': '语义模型',
            'agi_camera': '3D模块',
            'image_db': '图像数据库',
            'style_analyzer': '风格分析模型'
        }
        display_name = display_names.get(model_name, model_name)

        # 更新实例引用
        model = self.model_manager.get_model(model_name)
        if model_name == 'color_engine':
            self.color_engine = model
        elif model_name == 'nlp_parser':
            self.nlp_parser = model
        elif model_name == 'agi_camera':
            self.agi_camera = model
        elif model_name == 'image_db':
            self.image_db = model
            # 通知相关面板
            if hasattr(self, 'library_panel'):
                self.library_panel.set_database(self.image_db)
            if hasattr(self, 'agi_panel'):
                self.agi_panel.set_database(self.image_db)
        elif model_name == 'style_analyzer':
            self.style_analyzer = model

        self.statusbar.showMessage(f"{display_name}已加载", 2000)

    def _on_model_failed(self, model_name: str, error_msg: str):
        """模型加载失败时的回调"""
        QMessageBox.warning(self, "模型加载失败", f"加载 {model_name} 时出错:\n{error_msg}")
        self.statusbar.showMessage(f"加载 {model_name} 失败", 3000)

    def _ensure_model(self, model_name: str):
        """确保模型已加载（同步方式,会阻塞）"""
        if not self.model_manager.is_loaded(model_name):
            self.model_manager.load_model_sync(model_name)

        # 同步加载后手动更新实例引用(因为同步加载不会触发信号)
        model = self.model_manager.get_model(model_name)
        if model:
            if model_name == 'color_engine':
                self.color_engine = model
            elif model_name == 'nlp_parser':
                self.nlp_parser = model
            elif model_name == 'agi_camera':
                self.agi_camera = model
            elif model_name == 'image_db':
                self.image_db = model
            elif model_name == 'style_analyzer':
                self.style_analyzer = model

    def _ensure_model_async(self, model_name: str):
        """确保模型已加载（异步方式,不阻塞）"""
        if not self.model_manager.is_loaded(model_name):
            self.model_manager.load_model_async(model_name)

    def _deferred_init(self):
        """延迟初始化（窗口显示后执行）"""
        # 后台异步加载图像数据库（不阻塞UI）
        self.statusbar.showMessage("正在后台加载图像数据库...", 2000)
        self.model_manager.load_model_async('image_db')

    def _background_load_db(self):
        """后台加载数据库（已弃用,使用 ModelManager 替代）"""
        pass


    def _setup_ui(self):
        """设置UI布局"""
        self.setWindowTitle(UI_CONFIG["window_title"])
        self.resize(*UI_CONFIG["default_size"])
        self.setMinimumSize(*UI_CONFIG["min_size"])

        # 中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧: 图像查看器
        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)

        # 右侧: 功能面板
        right_panel = QTabWidget()
        right_panel.setMinimumWidth(450)  # 增大最小宽度以容纳更大的预览区域
        right_panel.setMaximumWidth(600)  # 增大最大宽度

        # 调色面板
        self.color_panel = ColorGradingPanel()
        right_panel.addTab(self.color_panel, "一句话调色")

        # AGI相机面板 - 添加滚动区域以支持更大的预览
        self.agi_panel = AGICameraPanel()
        agi_scroll = QScrollArea()
        agi_scroll.setWidgetResizable(True)
        agi_scroll.setWidget(self.agi_panel)
        agi_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        agi_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_panel.addTab(agi_scroll, "AGI相机")

        # 图像库面板 (数据库延迟加载)
        self.library_panel = ImageLibraryPanel()
        right_panel.addTab(self.library_panel, "图像库")

        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([800, 500])

        # 底部停靠窗口逻辑已移除

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")

        open_action = QAction("打开(&O)", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        open_folder_action = QAction("打开文件夹...", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        save_action = QAction("保存(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)

        save_as_action = QAction("另存为...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")

        undo_action = QAction("撤销(&Z)", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        reset_action = QAction("重置", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_image)
        edit_menu.addAction(reset_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")

        zoom_in_action = QAction("放大", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("缩小", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("适应窗口", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.image_viewer.fit_to_view)
        view_menu.addAction(fit_action)

        view_menu.addSeparator()
        
        manage_library_action = QAction("管理图像库...", self)
        manage_library_action.triggered.connect(self.open_library_manager)
        view_menu.addAction(manage_library_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 打开文件
        open_btn = QAction("打开", self)
        open_btn.setToolTip("打开图像 (Ctrl+O)")
        open_btn.triggered.connect(self.open_image)
        toolbar.addAction(open_btn)

        # 保存
        save_btn = QAction("保存", self)
        save_btn.setToolTip("保存图像 (Ctrl+S)")
        save_btn.triggered.connect(self.save_image)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        # 撤销
        undo_btn = QAction("撤销", self)
        undo_btn.setToolTip("撤销 (Ctrl+Z)")
        undo_btn.triggered.connect(self.undo)
        toolbar.addAction(undo_btn)

        # 重置
        reset_btn = QAction("重置", self)
        reset_btn.setToolTip("重置到原始图像 (Ctrl+R)")
        reset_btn.triggered.connect(self.reset_image)
        toolbar.addAction(reset_btn)

        toolbar.addSeparator()

        # 对比视图
        compare_btn = QAction("对比", self)
        compare_btn.setCheckable(True)
        compare_btn.setToolTip("显示前后对比")
        compare_btn.triggered.connect(self.toggle_compare)
        toolbar.addAction(compare_btn)
        self.compare_btn = compare_btn

    def _setup_statusbar(self):
        """设置状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # 图像信息标签
        self.image_info_label = QLabel("未加载图像")
        self.statusbar.addWidget(self.image_info_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.statusbar.addPermanentWidget(self.progress_bar)

    def _connect_signals(self):
        """连接信号"""
        # 调色面板信号
        self.color_panel.params_changed.connect(self.apply_color_grading)
        self.color_panel.text_input_submitted.connect(self.process_text_command)
        self.color_panel.find_similar_requested.connect(self.find_similar_images)
        self.color_panel.upload_reference_requested.connect(self.upload_and_apply_reference)

        # AGI相机信号
        self.agi_panel.generate_3d_requested.connect(self.generate_3d)
        self.agi_panel.generate_animation_requested.connect(self.generate_animation)
        self.agi_panel.object_selected.connect(self._on_object_selected)
        self.agi_panel.object_box_selected.connect(self._on_object_box_selected)
        self.agi_panel.object_path_selected.connect(self._on_object_path_selected)
        self.agi_panel.generate_object_3d_requested.connect(self._on_generate_object_3d)
        self.agi_panel.generate_multiview_3d_requested.connect(self._on_generate_multiview_3d)
        self.agi_panel.export_3d_model_requested.connect(self.export_3d_model)

        # 图像库信号
        # 图像库信号
        self.library_panel.image_selected.connect(self.load_reference_image)
        self.library_panel.import_requested.connect(self.import_images)


    def open_image(self):
        """打开图像文件"""
        # 使用统一的图像选择对话框
        paths = pick_images(self.image_db, multi_select=False, parent=self)

        if paths and len(paths) > 0:
            self.load_image(paths[0])

    def open_folder(self):
        """打开文件夹并索引图像"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")

        if folder_path:
            self.index_folder(folder_path)

    def load_image(self, file_path: str):
        """加载图像"""
        try:
            image = imread_safe(file_path)
            if image is None:
                raise ValueError("无法读取图像文件")

            self.current_image = image.copy()
            self.original_image = image.copy()
            self.current_file_path = file_path

            # 清空历史记录（新图像）
            self._clear_history()

            # 更新显示
            self.update_image_display()

            # 更新状态栏
            h, w = image.shape[:2]
            self.image_info_label.setText(f"{Path(file_path).name} | {w}x{h}")

            # 重置调色面板
            self.color_panel.reset_params()

            # 设置图像到AGI相机（仅当已加载时）
            self.agi_panel.set_image(image)
            if self.agi_camera is not None:
                self.agi_camera.set_image(image)


        except (ValueError, OSError, IOError) as e:
            # 捕获文件读取和图像处理相关的错误
            QMessageBox.critical(self, "错误", f"加载图像失败: {e}")

    def update_image_display(self):
        """更新图像显示"""
        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)

    def save_image(self):
        """保存图像"""
        if self.current_image is None:
            return

        if self.current_file_path:
            imwrite_safe(self.current_file_path, self.current_image)
            self.statusbar.showMessage("图像已保存", 3000)
        else:
            self.save_image_as()

    def save_image_as(self):
        """另存为"""
        if self.current_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "",
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;所有文件 (*.*)"
        )

        if file_path:
            imwrite_safe(file_path, self.current_image)
            self.current_file_path = file_path
            self.statusbar.showMessage("图像已保存", 3000)

    def undo(self):
        """撤销操作 - 返回上一步状态"""
        if not self._history_stack:
            # 没有历史记录，恢复到原始图像
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
                self.update_image_display()
                self.color_panel.reset_params()
                self._current_params = None
                self.statusbar.showMessage("已恢复到原始图像", 2000)
            return

        # 弹出上一步状态
        prev_state = self._history_stack.pop()
        self.current_image = prev_state['image'].copy()
        self._current_params = prev_state['params']

        # 更新显示
        self.update_image_display()

        # 更新调色面板参数
        if self._current_params is not None:
            self.color_panel.set_params(self._current_params)
        else:
            self.color_panel.reset_params()

        self.statusbar.showMessage(f"已撤销 (剩余 {len(self._history_stack)} 步)", 2000)

    def _save_history(self):
        """保存当前状态到历史记录"""
        if self.current_image is None:
            return

        # 保存当前状态
        state = {
            'image': self.current_image.copy(),
            'params': self._current_params
        }
        self._history_stack.append(state)

        # 限制历史记录数量
        if len(self._history_stack) > self.MAX_HISTORY:
            self._history_stack.pop(0)

    def _clear_history(self):
        """清空历史记录"""
        self._history_stack.clear()
        self._current_params = None

    def reset_image(self):
        """重置图像"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_image_display()
            self.color_panel.reset_params()
            # 清空历史记录
            self._clear_history()

    def toggle_compare(self, checked: bool):
        """切换对比视图"""
        if checked and self.original_image is not None:
            self.image_viewer.set_compare_mode(self.original_image, self.current_image)
        else:
            self.image_viewer.set_compare_mode(None, None)

    # def toggle_library(self, checked: bool): # 移除旧的切换方法
    #     """切换图像库显示"""
    #     if checked:
    #         self.library_dock.show()
    #     else:
    #         self.library_dock.hide()

    def _wait_for_thread(self):
        """
        等待当前线程完成

        安全地等待工作线程结束,不使用危险的terminate()
        如果线程无法正常结束,记录警告但不强制终止
        """
        if self.thread is not None and self.thread.isRunning():
            print("[MainWindow] 等待旧线程完成...")
            self.thread.request_stop()  # 请求停止

            # 增加等待时间到10秒,给线程足够时间清理资源
            if not self.thread.wait(10000):  # 最多等待10秒
                print("[MainWindow] 警告: 线程等待超时")
                print("[MainWindow] 线程仍在运行,将在后台继续执行")
                # 不使用terminate(),让线程自然结束
                # 如果线程持有资源,强制终止会导致资源泄漏或死锁

                # 标记线程为孤立状态,不再管理它
                self.thread.setParent(None)
                self.thread = None
                return

            self.thread.deleteLater()  # 标记为稍后删除
            self.thread = None

    def apply_color_grading(self, params):
        """应用调色参数"""
        if self.original_image is None:
            return

        # 保存当前状态到历史记录（在应用新调色前）
        self._save_history()

        # 懒加载调色引擎
        self._ensure_model('color_engine')

        # 保存待应用的参数，用于完成后更新
        self._pending_params = params

        self._wait_for_thread()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # 不确定进度

        def process():
            return self.color_engine.apply_grading(self.original_image, params)

        self.thread = ProcessingThread(process)
        self.thread.finished.connect(self._on_grading_finished)
        self.thread.error.connect(self._on_processing_error)
        self.thread.start()

    def _on_grading_finished(self, result):
        """调色完成回调"""
        self.progress_bar.hide()
        self.current_image = result
        self.update_image_display()

        # 更新当前参数（用于撤销时恢复）
        if hasattr(self, '_pending_params'):
            self._current_params = self._pending_params
            self._pending_params = None

    def _on_processing_error(self, error_msg):
        """处理错误回调"""
        self.progress_bar.hide()
        QMessageBox.warning(self, "处理错误", error_msg)

    def process_text_command(self, text: str):
        """处理文本命令（异步）"""
        if self.original_image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        # 懒加载NLP解析器和调色引擎
        self._ensure_model('nlp_parser')
        self._ensure_model('color_engine')

        # 状态反馈
        self.statusbar.showMessage(f"正在分析调色指令: {text[:30]}...")
        print(f"[调色指令] 处理: {text}")

        # 显示进度指示器
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # 不确定进度

        # 禁用应用按钮，防止重复提交
        if hasattr(self, 'color_panel'):
            self.color_panel.apply_btn.setEnabled(False)
            self.color_panel.apply_btn.setText("分析中...")

        # 检查是否需要查找参考图像
        reference_params = None
        if "复刻" in text or "参考" in text:
            # 尝试从文本中提取搜索关键词
            if self.image_db:
                results = self.image_db.search_by_text(text, top_k=1)
                if results:
                    ref_path = results[0]["path"]
                    ref_image = imread_safe(ref_path)
                    if ref_image is not None:
                        reference_params = self.color_engine.extract_color_params(ref_image)
                        print(f"[调色指令] 使用参考图片: {ref_path}")

        # 定义成功回调
        def on_parse_success(params):
            """解析成功回调"""
            # 打印解析结果
            print(f"[调色指令] 解析结果: 曝光={params.exposure:.2f}, 对比度={params.contrast:.2f}, "
                  f"色温={params.temperature:.0f}, 饱和度={params.saturation:.2f}")

            # 隐藏进度条
            self.progress_bar.hide()

            # 恢复按钮状态
            if hasattr(self, 'color_panel'):
                self.color_panel.apply_btn.setEnabled(True)
                self.color_panel.apply_btn.setText("应用")

            # 更新面板显示
            self.color_panel.set_params(params)

            # 应用调色
            self.apply_color_grading(params)

            self.statusbar.showMessage("调色指令已应用", 3000)

        # 定义错误回调
        def on_parse_error(error_msg: str):
            """解析失败回调"""
            self.progress_bar.hide()

            # 恢复按钮状态
            if hasattr(self, 'color_panel'):
                self.color_panel.apply_btn.setEnabled(True)
                self.color_panel.apply_btn.setText("应用")

            self.statusbar.showMessage(f"分析失败: {error_msg}", 5000)
            QMessageBox.warning(self, "分析失败", f"无法分析调色指令:\n{error_msg}")

        # 异步解析
        self.nlp_parser.parse_async(
            text,
            on_success=on_parse_success,
            on_error=on_parse_error,
            reference_params=reference_params
        )

    def import_images(self, file_paths: list, group: str = "默认"):
        """导入图像到库"""
        if not file_paths:
            return

        self._ensure_model('image_db')
        self._wait_for_thread()
        
        self.progress_bar.show()
        self.progress_bar.setRange(0, len(file_paths))
        self.statusbar.showMessage(f"准备导入 {len(file_paths)} 张图片到分组 '{group}'...")

        def process_import():
            success_count = 0
            for i, path in enumerate(file_paths):
                try:
                    self.image_db.add_image(path, group=group)
                    success_count += 1
                except (OSError, ValueError, RuntimeError) as e:
                    # 捕获文件和数据库相关错误
                    print(f"导入失败 {path}: {e}")
                
                if self.thread:
                    self.thread.progress.emit(i + 1)
            return success_count

        self.thread = ProcessingThread(process_import)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(lambda n: self._on_import_finished(n))
        self.thread.start()

    def open_library_manager(self):
        """打开图像库管理器"""
        self._ensure_model('image_db')
        dialog = LibraryManagerDialog(self.image_db, self)
        # 连接导入信号
        dialog.import_requested.connect(self.import_images)
        dialog.exec_()
        # 关闭后刷新面板
        self.library_panel.refresh()

    def _on_import_finished(self, count: int):
        """导入完成回调"""
        self.progress_bar.hide()
        self.library_panel.refresh()
        self.statusbar.showMessage(f"成功导入 {count} 张图片", 3000)
        # 如果管理器打开着，它也许需要刷新，但这是一个模态对话框，通常关闭后刷新即可
        # 或者可以发送信号通知
        QMessageBox.information(self, "导入完成", f"已成功导入 {count} 张图片到图像库")

    def find_similar_images(self):
        """查找相似风格图像（混合搜索）"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return
            
        # 1. 风格分析
        self._ensure_model('style_analyzer')
        self.statusbar.showMessage("正在分析图像风格...")
        
        try:
            summary = self.style_analyzer.analyze(self.current_image)
            
            # 显示风格总结
            summary_text = summary.to_string()
            print(f"[风格分析]\n{summary_text}")
            self.statusbar.showMessage(f"风格分析: {summary.style_tags[:2]}", 3000)
            
            # 2. 本地搜索
            self._ensure_model('image_db') 
            local_results = self.image_db.search_similar(self.current_image, top_k=10)
            
            # 3. 网络搜索
            web_message = "网络搜索: 未连接"
            web_results = []
            
            if self._check_internet_connection():
                self.statusbar.showMessage("正在进行网络搜索...")
                web_results = self._search_web_for_style(summary)
                web_message = f"网络搜索: 找到 {len(web_results)} 个结果"
            else:
                web_message = "网络搜索: No Internet"
                
            # 合并结果展示
            self.library_panel.show_search_results(local_results)
            # 切换到图像库Tab (假设是index 2)
            # 查找 TabWidget
            tab_widget = self.library_panel.parent().parent() # QTabWidget -> QStackWidget -> Panel
            if isinstance(tab_widget, QTabWidget):
                # 遍历找到 library_panel
                idx = tab_widget.indexOf(self.library_panel)
                if idx >= 0:
                    tab_widget.setCurrentIndex(idx)
            
            # 弹窗显示完整报告
            msg = f"""<b>风格分析报告:</b><br/>
            风格标签: {', '.join(summary.style_tags)}<br/>
            内容标签: {', '.join(summary.content_tags)}<br/>
            画面结构: {summary.structure['orientation']} ({summary.structure['composition_guess']})<br/>
            <hr/>
            <b>查找结果:</b><br/>
            本地库: 找到 {len(local_results)} 张<br/>
            {web_message}
            """
            
            if web_results:
                msg += "<br/><br/><b>网络资源推荐 (点击链接搜索):</b><br/>"
                search_query = f"{' '.join(summary.style_tags)} {' '.join(summary.content_tags)} photography"
                import urllib.parse
                url = f"https://unsplash.com/s/photos/{urllib.parse.quote(search_query)}"
                msg += f'<a href="{url}">在 Unsplash 上搜索 "{search_query}"</a>'
            
            mbox = QMessageBox(self)
            mbox.setWindowTitle("相似风格查找")
            mbox.setTextFormat(Qt.RichText)
            mbox.setText(msg)
            mbox.exec_()
            
        except (RuntimeError, ValueError, ImportError, AttributeError) as e:
            # 捕获模型和分析相关错误
            QMessageBox.critical(self, "错误", f"风格分析失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _check_internet_connection(self) -> bool:
        """检查网络连接"""
        import socket
        try:
            # 尝试连接常用的DNS服务器
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except OSError:
            return False

    def _search_web_for_style(self, summary) -> list:
        """
        模拟网络搜索
        实际上我们会生成搜索关键词，并返回一些模拟或通过简单API获取的结果
        """
        # 构建搜索词
        keywords = summary.style_tags[:2] + summary.content_tags[:2]
        query = " ".join(keywords)
        print(f"[网络搜索] Keywords: {query}")
        
        # 这里模拟返回一些结果，或者如果未来集成了爬虫可以在这里实现
        # 为了演示，我们返回一个非空列表表示"可以搜索"
        return [{"title": f"Web Result for {query}", "url": "..."}]


    def load_reference_image(self, image_path: str):
        """加载参考图像并在左侧显示预览"""
        # 读取图片并在左侧 ImageViewer 中显示
        ref_image = imread_safe(image_path)
        if ref_image is not None:
            # 在左侧预览显示图片
            self.current_image = ref_image.copy()
            self.original_image = ref_image.copy()
            self.current_file_path = image_path
            self.update_image_display()

            # 更新状态栏
            h, w = ref_image.shape[:2]
            self.image_info_label.setText(f"{Path(image_path).name} | {w}x{h}")

            # 设置到 AGI 面板
            self.agi_panel.set_image(ref_image)
            if self.agi_camera is not None:
                self.agi_camera.set_image(ref_image)

            # 重置调色面板
            self.color_panel.reset_params()

            self.statusbar.showMessage(f"已加载图片: {Path(image_path).name}", 3000)
        else:
            self.statusbar.showMessage(f"无法加载图片: {image_path}", 3000)
    
    def upload_and_apply_reference(self):
        """上传参考图片并提取色调应用到当前图片"""
        if self.original_image is None:
            QMessageBox.warning(self, "提示", "请先加载要调整的图片")
            return

        # 使用统一的图像选择对话框
        paths = pick_images(self.image_db, multi_select=False, parent=self)

        if not paths or len(paths) == 0:
            return

        file_path = paths[0]

        # 读取参考图片
        ref_image = imread_safe(file_path)
        if ref_image is None:
            QMessageBox.warning(self, "错误", "无法读取选择的图片")
            self.color_panel.set_reference_status("读取参考图片失败", False)
            return
        
        # 更新状态
        from pathlib import Path as PathLib
        self.color_panel.set_reference_status(f"正在分析: {PathLib(file_path).name}...", True)
        self.statusbar.showMessage("正在提取参考图片色调...")
        
        # 提取色调参数
        try:
            params = self.color_engine.extract_color_params(ref_image)
            
            # 更新面板
            self.color_panel.set_params(params)
            
            # 应用到当前图片
            self.apply_color_grading(params)
            
            # 更新状态
            self.color_panel.set_reference_status(
                f"已应用参考图片色调: {PathLib(file_path).name}", True
            )
            self.statusbar.showMessage("参考图片色调已应用", 3000)
            
        except (ValueError, RuntimeError, OSError) as e:
            # 捕获图像处理和模型相关错误
            self.color_panel.set_reference_status(f"提取色调失败: {str(e)}", False)
            QMessageBox.warning(self, "错误", f"提取色调失败: {e}")

    def index_folder(self, folder_path: str):
        """索引文件夹中的图像"""
        folder = Path(folder_path)
        image_files = []
        for fmt in SUPPORTED_FORMATS:
            image_files.extend(folder.glob(f"*{fmt}"))
            image_files.extend(folder.glob(f"*{fmt.upper()}"))

        if not image_files:
            QMessageBox.information(self, "提示", "文件夹中没有支持的图像文件")
            return

        self._wait_for_thread()
        self.progress_bar.show()
        self.progress_bar.setRange(0, len(image_files))

        def index_images():
            for i, img_path in enumerate(image_files):
                try:
                    self.image_db.add_image(str(img_path))
                except Exception:
                    pass
                self.thread.progress.emit(i + 1)
            return len(image_files)

        self.thread = ProcessingThread(index_images)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(lambda n: self._on_index_finished(n))
        self.thread.start()

    def _on_index_finished(self, count: int):
        """索引完成回调"""
        self.progress_bar.hide()
        self.library_panel.refresh()
        self.statusbar.showMessage(f"已索引 {count} 张图像", 3000)

    def generate_3d(self, params: dict):
        """生成3D模型"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        # 懒加载AGI相机
        self._ensure_model('agi_camera')
        # 设置当前图像
        self.agi_camera.set_image(self.current_image)

        self._wait_for_thread()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)

        def generate():
            return self.agi_camera.capture_to_3d(
                self.current_image,
                depth_scale=params.get('depth_scale', 0.5),
                mesh_resolution=params.get('resolution', 256)
            )

        self.thread = ProcessingThread(generate)
        self.thread.finished.connect(self._on_3d_generated)
        self.thread.error.connect(self._on_processing_error)
        self.thread.start()

    def _on_3d_generated(self, mesh):
        """3D生成完成回调"""
        self.progress_bar.hide()
        self.agi_panel.set_mesh(mesh)
        self.statusbar.showMessage("3D模型生成完成", 3000)

    def generate_animation(self, params: dict):
        """生成动画"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先加载图像")
            return

        # 懒加载AGI相机
        self._ensure_model('agi_camera')
        self.agi_camera.set_image(self.current_image)

        self._wait_for_thread()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)

        def generate():
            return self.agi_camera.generate_demo_animation(
                self.current_image,
                num_frames=params.get('frames', 60),
                rotation_axis=params.get('axis', 'y'),
                depth_scale=params.get('depth_scale', 0.5)
            )

        self.thread = ProcessingThread(generate)
        self.thread.finished.connect(self._on_animation_generated)
        self.thread.error.connect(self._on_processing_error)
        self.thread.start()

    def _on_animation_generated(self, frames):
        """动画生成完成回调"""
        self.progress_bar.hide()
        self.agi_panel.set_animation(frames)
        self.statusbar.showMessage("动画生成完成", 3000)

    def export_3d_model(self):
        """导出3D模型"""
        mesh = self.agi_panel.get_current_mesh()
        if mesh is None:
            QMessageBox.warning(self, "提示", "请先生成3D模型")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出3D模型", "",
            "OBJ (*.obj);;GLTF (*.gltf);;GLB (*.glb);;STL (*.stl)"
        )

        if file_path:
            ext = Path(file_path).suffix.lower()[1:]
            self.agi_camera.export_3d_model(mesh, file_path, ext)
            self.statusbar.showMessage(f"3D模型已导出: {file_path}", 3000)

    def _on_object_selected(self, x: int, y: int):
        """处理物体点选（异步执行避免UI卡顿）"""
        print(f"[MainWindow] 收到点选信号: ({x}, {y})")
        if self.current_image is None:
            print("[MainWindow] 当前没有图像")
            return

        # 懒加载AGI相机
        self._ensure_model('agi_camera')
        self.agi_camera.set_image(self.current_image)

        self.statusbar.showMessage(f"正在分割物体... 点击位置: ({x}, {y})")
        QApplication.processEvents()  # 更新UI

        # 异步执行分割
        def do_segment():
            return self.agi_camera.select_object_at_point(x, y)

        self._wait_for_thread()
        self.thread = ProcessingThread(do_segment)
        self.thread.finished.connect(self._on_segment_finished)
        self.thread.error.connect(lambda e: self._on_segment_error(e))
        self.thread.start()

    def _on_object_box_selected(self, x1: int, y1: int, x2: int, y2: int):
        """处理物体框选（异步执行避免UI卡顿）"""
        print(f"[MainWindow] 收到框选信号: ({x1}, {y1}) - ({x2}, {y2})")
        if self.current_image is None:
            print("[MainWindow] 当前没有图像")
            return

        # 懒加载AGI相机
        self._ensure_model('agi_camera')
        self.agi_camera.set_image(self.current_image)

        self.statusbar.showMessage(f"正在分割物体... 框选区域: ({x1}, {y1}) - ({x2}, {y2})")
        QApplication.processEvents()  # 更新UI

        # 异步执行分割
        def do_segment():
            return self.agi_camera.select_object_with_box(x1, y1, x2, y2)

        self._wait_for_thread()
        self.thread = ProcessingThread(do_segment)
        self.thread.finished.connect(self._on_segment_finished)
        self.thread.error.connect(lambda e: self._on_segment_error(e))
        self.thread.start()

    def _on_object_path_selected(self, path_points: list):
        """处理物体划线选择（异步执行避免UI卡顿）"""
        print(f"[MainWindow] 收到划线路径信号: {len(path_points)} 个点")
        if self.current_image is None:
            print("[MainWindow] 当前没有图像")
            return

        # 懒加载AGI相机
        self._ensure_model('agi_camera')
        self.agi_camera.set_image(self.current_image)

        self.statusbar.showMessage(f"正在分割物体... 划线路径: {len(path_points)} 个点")
        QApplication.processEvents()  # 更新UI

        # 异步执行分割
        def do_segment():
            return self.agi_camera.select_object_with_path(path_points)

        self._wait_for_thread()
        self.thread = ProcessingThread(do_segment)
        self.thread.finished.connect(self._on_segment_finished)
        self.thread.error.connect(lambda e: self._on_segment_error(e))
        self.thread.start()

    def _on_segment_finished(self, mask):
        """分割完成回调"""
        print(f"[MainWindow] 分割结果: {'成功' if mask is not None else '失败'}")

        if mask is not None:
            # 更新预览
            overlay = self.agi_camera.get_mask_overlay()
            if overlay is not None:
                self.agi_panel.update_selection_preview(overlay)
            self.agi_panel.set_selection_result(True, "物体已选中")
            self.statusbar.showMessage("物体分割完成", 3000)
        else:
            self.agi_panel.set_selection_result(False, "分割失败，请重试")
            self.statusbar.showMessage("物体分割失败", 3000)

    def _on_segment_error(self, error_msg: str):
        """分割错误回调"""
        print(f"[MainWindow] 分割错误: {error_msg}")
        self.agi_panel.set_selection_result(False, f"分割失败: {error_msg}")
        self.statusbar.showMessage("物体分割失败", 3000)

    def _on_generate_object_3d(self, params: dict):
        """从选中物体生成3D"""
        # 懒加载AGI相机（如果还未加载）
        self._ensure_model('agi_camera')
        
        if not self.agi_camera.has_selection():
            QMessageBox.warning(self, "提示", "请先选择物体")
            return

        self.progress_bar.show()
        self.statusbar.showMessage("正在生成选中物体的3D动画...")

        def generate():
            return self.agi_camera.generate_object_3d_animation(
                num_frames=params.get('frames', 60),
                rotation_axis=params.get('axis', 'y'),
                depth_scale=params.get('depth_scale', 0.5)
            )

        self.thread = ProcessingThread(generate)
        self.thread.finished.connect(self._on_animation_generated)
        self.thread.error.connect(self._on_processing_error)
        self.thread.start()

    def _on_generate_multiview_3d(self, params: dict):
        """从多视角图片生成3D模型"""
        image_paths = params.get('image_paths', [])
        if len(image_paths) < 2:
            QMessageBox.warning(self, "提示", "请至少选择2张不同角度的图片")
            return

        self._wait_for_thread()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self.statusbar.showMessage(f"正在从{len(image_paths)}张图片进行多视角3D重建...")

        def generate():
            mesh, frames = self.agi_camera.generate_multiview_3d(
                image_paths,
                depth_scale=params.get('depth_scale', 0.7),
                mesh_resolution=params.get('resolution', 256)
            )
            return {'mesh': mesh, 'frames': frames}

        self.thread = ProcessingThread(generate)
        self.thread.finished.connect(self._on_multiview_3d_generated)
        self.thread.error.connect(self._on_processing_error)
        self.thread.start()

    def _on_multiview_3d_generated(self, result):
        """多视角3D生成完成回调"""
        self.progress_bar.hide()
        mesh = result.get('mesh')
        frames = result.get('frames')
        if mesh:
            self.agi_panel.set_mesh(mesh)
        if frames:
            self.agi_panel.set_animation(frames)
        self.statusbar.showMessage("多视角3D重建完成", 3000)

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 AI全模态影像处理",
            f"""<h3>AI全模态影像处理 v{APP_VERSION}</h3>
            <p>一款基于AI的智能图像处理软件</p>
            <p><b>核心功能:</b></p>
            <ul>
                <li>一句话调色 - 自然语言智能调色</li>
                <li>图像特征检索 - 相似风格匹配</li>
                <li>AGI相机 - 3D演示动画生成</li>
            </ul>
            <p>基于端侧低bit量化技术,支持快速推理</p>
            """
        )

    def closeEvent(self, event):
        """
        关闭事件 - 确保线程正确停止

        安全地关闭应用程序,不使用危险的terminate()
        """
        print("[MainWindow] 正在关闭窗口...")

        # 停止图像处理线程
        if self.thread is not None and self.thread.isRunning():
            print("[MainWindow] 请求停止处理线程...")
            self.thread.request_stop()
            if not self.thread.wait(5000):  # 等待最多5秒
                print("[MainWindow] 警告: 处理线程等待超时")
                print("[MainWindow] 将在后台等待线程完成")
                # 不使用terminate(),让线程自然结束
                # 应用程序关闭时线程会自动结束
            else:
                print("[MainWindow] 处理线程已停止")

        # 取消正在进行的LLM分析
        if self.nlp_parser is not None and hasattr(self.nlp_parser, 'llm_analyzer'):
            if self.nlp_parser.llm_analyzer is not None:
                try:
                    print("[MainWindow] 取消LLM分析...")
                    self.nlp_parser.llm_analyzer.cancel_current()
                except (RuntimeError, AttributeError) as e:
                    # 捕获线程清理相关错误
                    print(f"清理LLM线程时出错: {e}")

        print("[MainWindow] 窗口关闭完成")
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setApplicationName("AI全模态影像处理")

    # 设置应用程序默认字体，避免QFont警告
    default_font = QFont()
    default_font.setFamily("Microsoft YaHei")  # 使用微软雅黑
    default_font.setPointSize(10)  # 设置有效的字体大小
    app.setFont(default_font)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
