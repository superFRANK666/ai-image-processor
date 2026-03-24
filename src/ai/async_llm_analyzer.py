"""
异步LLM调色分析器
在后台线程中执行推理，避免阻塞UI
"""
from typing import Dict, Any, Optional, Callable
from PySide6.QtCore import QThread, Signal
import threading


class AsyncLLMAnalysisThread(QThread):
    """异步LLM分析线程"""

    # 信号：分析完成
    analysis_finished = Signal(dict)  # 返回分析结果
    # 信号：分析失败
    analysis_failed = Signal(str)  # 返回错误信息

    def __init__(self, llm_analyzer, description: str):
        super().__init__()
        self.llm_analyzer = llm_analyzer
        self.description = description
        self._stop_requested = False

    def run(self):
        """在后台线程中执行分析"""
        try:
            result = self.llm_analyzer.analyze(self.description)
            if not self._stop_requested:
                self.analysis_finished.emit(result)
        except Exception as e:
            if not self._stop_requested:
                self.analysis_failed.emit(str(e))
    
    def request_stop(self):
        """请求线程停止"""
        self._stop_requested = True


class AsyncLLMColorAnalyzer:
    """异步LLM调色分析器包装器"""

    def __init__(self, llm_analyzer):
        """
        初始化异步分析器

        Args:
            llm_analyzer: 底层的LLM分析器实例（LocalLLMColorAnalyzer）
        """
        self.llm_analyzer = llm_analyzer
        self._current_thread: Optional[AsyncLLMAnalysisThread] = None
        self._lock = threading.Lock()

    def analyze_async(
        self,
        description: str,
        on_success: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ) -> AsyncLLMAnalysisThread:
        """
        异步分析用户描述

        Args:
            description: 用户描述
            on_success: 成功回调函数
            on_error: 错误回调函数

        Returns:
            分析线程对象
        """
        # 如果有正在运行的线程，取消它
        with self._lock:
            if self._current_thread and self._current_thread.isRunning():
                print("[AsyncLLMColorAnalyzer] 取消旧分析线程...")
                self._current_thread.request_stop()
                if not self._current_thread.wait(3000):
                    print("[AsyncLLMColorAnalyzer] 警告: 线程等待超时，强制终止")
                    self._current_thread.terminate()
                    self._current_thread.wait()
                self._current_thread.deleteLater()
                self._current_thread = None

        # 创建新线程
        thread = AsyncLLMAnalysisThread(self.llm_analyzer, description)

        # 连接信号
        if on_success:
            thread.analysis_finished.connect(on_success)
        if on_error:
            thread.analysis_failed.connect(on_error)

        # 保存当前线程引用
        with self._lock:
            self._current_thread = thread

        # 启动线程
        thread.start()

        return thread

    def analyze_sync(self, description: str) -> Dict[str, Any]:
        """
        同步分析（阻塞调用，用于兼容旧代码）

        Args:
            description: 用户描述

        Returns:
            分析结果
        """
        return self.llm_analyzer.analyze(description)

    def cancel_current(self):
        """取消当前正在进行的分析"""
        with self._lock:
            if self._current_thread and self._current_thread.isRunning():
                print("[AsyncLLMColorAnalyzer] 取消当前分析...")
                self._current_thread.request_stop()
                if not self._current_thread.wait(3000):
                    print("[AsyncLLMColorAnalyzer] 警告: 取消等待超时，强制终止")
                    self._current_thread.terminate()
                    self._current_thread.wait()
                self._current_thread.deleteLater()
                self._current_thread = None

    def is_busy(self) -> bool:
        """检查是否有分析正在进行"""
        with self._lock:
            return self._current_thread is not None and self._current_thread.isRunning()

