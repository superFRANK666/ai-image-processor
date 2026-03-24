"""
核心模块 - 模型管理器
统一管理AI模型的生命周期
"""
from PySide6.QtCore import QObject, Signal, QThread
import threading
from typing import Dict, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoadWorker(QThread):
    """模型加载工作线程"""
    loaded = Signal(str, object)  # (model_name, model_instance)
    failed = Signal(str, str)     # (model_name, error_message)

    def __init__(self, model_name: str, load_func: Callable):
        super().__init__()
        self.model_name = model_name
        self.load_func = load_func

    def run(self):
        try:
            logger.info(f"开始加载模型: {self.model_name}")
            model = self.load_func()
            self.loaded.emit(self.model_name, model)
            logger.info(f"模型加载成功: {self.model_name}")
        except Exception as e:
            error_msg = f"加载模型 {self.model_name} 失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.failed.emit(self.model_name, error_msg)


class ModelManager(QObject):
    """
    模型管理器

    统一管理所有AI模型的加载、缓存和生命周期
    支持异步加载,避免阻塞UI线程

    功能:
    - 延迟加载: 只在需要时加载模型
    - 异步加载: 在后台线程加载,不阻塞UI
    - 缓存管理: 避免重复加载同一模型
    - 事件通知: 加载完成时发出信号
    """

    model_loaded = Signal(str)   # 模型加载完成信号
    model_failed = Signal(str, str)  # 模型加载失败信号
    loading_started = Signal(str)  # 开始加载信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: Dict[str, Any] = {}  # 已加载的模型实例
        self._load_functions: Dict[str, Callable] = {}  # 模型加载函数
        self._loading: Dict[str, bool] = {}  # 正在加载的模型
        self._workers: Dict[str, ModelLoadWorker] = {}  # 工作线程

    def register_model(self, model_name: str, load_func: Callable):
        """
        注册模型及其加载函数

        Args:
            model_name: 模型名称(唯一标识)
            load_func: 加载模型的函数,返回模型实例
        """
        self._load_functions[model_name] = load_func
        logger.debug(f"注册模型: {model_name}")

    def get_model(self, model_name: str) -> Optional[Any]:
        """
        获取已加载的模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例,如果未加载则返回None
        """
        return self._models.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self._models

    def is_loading(self, model_name: str) -> bool:
        """检查模型是否正在加载"""
        return self._loading.get(model_name, False)

    def load_model_async(self, model_name: str):
        """
        异步加载模型(在后台线程中)

        Args:
            model_name: 模型名称

        注意: 此方法立即返回,加载完成后会发出 model_loaded 信号
        """
        # 如果已加载,直接发出信号
        if self.is_loaded(model_name):
            self.model_loaded.emit(model_name)
            return

        # 如果正在加载,不重复加载
        if self.is_loading(model_name):
            logger.debug(f"模型 {model_name} 正在加载中,跳过")
            return

        # 检查是否已注册
        if model_name not in self._load_functions:
            error_msg = f"模型 {model_name} 未注册"
            logger.error(error_msg)
            self.model_failed.emit(model_name, error_msg)
            return

        # 开始异步加载
        self._loading[model_name] = True
        self.loading_started.emit(model_name)

        worker = ModelLoadWorker(model_name, self._load_functions[model_name])
        worker.loaded.connect(self._on_model_loaded)
        worker.failed.connect(self._on_model_failed)
        worker.finished.connect(lambda: self._cleanup_worker(model_name))

        self._workers[model_name] = worker
        worker.start()

    def load_model_sync(self, model_name: str) -> Optional[Any]:
        """
        同步加载模型(阻塞当前线程)

        Args:
            model_name: 模型名称

        Returns:
            模型实例,失败返回None

        警告: 此方法会阻塞UI线程,仅在必要时使用
        """
        # 如果已加载,直接返回
        if self.is_loaded(model_name):
            return self._models[model_name]

        # 检查是否已注册
        if model_name not in self._load_functions:
            logger.error(f"模型 {model_name} 未注册")
            return None

        try:
            logger.info(f"同步加载模型: {model_name}")
            model = self._load_functions[model_name]()
            self._models[model_name] = model
            logger.info(f"模型加载成功: {model_name}")
            return model
        except Exception as e:
            logger.error(f"同步加载模型 {model_name} 失败: {e}", exc_info=True)
            return None

    def unload_model(self, model_name: str):
        """
        卸载模型,释放内存

        Args:
            model_name: 模型名称
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"卸载模型: {model_name}")

    def unload_all(self):
        """卸载所有模型"""
        self._models.clear()
        logger.info("卸载所有模型")

    def _on_model_loaded(self, model_name: str, model: Any):
        """模型加载完成回调"""
        self._models[model_name] = model
        self._loading[model_name] = False
        self.model_loaded.emit(model_name)

    def _on_model_failed(self, model_name: str, error_msg: str):
        """模型加载失败回调"""
        self._loading[model_name] = False
        self.model_failed.emit(model_name, error_msg)

    def _cleanup_worker(self, model_name: str):
        """清理工作线程"""
        if model_name in self._workers:
            worker = self._workers[model_name]
            worker.deleteLater()
            del self._workers[model_name]

    def get_loaded_models(self) -> list:
        """获取所有已加载的模型名称"""
        return list(self._models.keys())

