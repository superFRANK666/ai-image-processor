"""
图像库管理对话框
用于管理、浏览、删除和导入图像库资源
"""
import logging
import os
import platform
import subprocess
from pathlib import Path
import cv2

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QSplitter, QGroupBox, QFormLayout, QMenu, QFrame
)
from PySide6.QtCore import Qt, Signal, QSize, QThread
from PySide6.QtGui import QIcon, QPixmap, QImage

# 使用相对导入项目模块
from ..ai import ImageIndexDatabase
from .image_picker_dialog import pick_images
from .ui_utils import fit_thumbnail_size


logger = logging.getLogger(__name__)


class ThumbnailLoader(QThread):
    """后台加载缩略图线程"""
    thumbnail_loaded = Signal(str, QImage)  # id, image

    def __init__(self, images_data, icon_size=120):
        super().__init__()
        self.images_data = images_data
        self.icon_size = icon_size
        self._is_running = True

    def run(self):
        for img_data in self.images_data:
            if not self._is_running:
                break
            
            img_id = img_data['id']
            path = img_data['path']
            
            try:
                # 简单缓存检查（这里暂不实现复杂缓存，直接加载）
                image = imread_safe(path)
                if image is not None:
                    # 调整大小
                    h, w = image.shape[:2]
                    new_w, new_h = fit_thumbnail_size(w, h, self.icon_size)
                    image = cv2.resize(image, (new_w, new_h))
                    
                    # 转 RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    self.thumbnail_loaded.emit(img_id, q_image)
            except Exception as exc:
                logger.debug("Failed to load thumbnail for %s: %s", path, exc)
                
    def stop(self):
        self._is_running = False

class LibraryManagerDialog(QDialog):
    """图像库管理器"""
    
    import_requested = Signal(list)  # 请求导入
    delete_requested = Signal(str)   # 请求删除
    
    def __init__(self, image_db: ImageIndexDatabase, parent=None):
        super().__init__(parent)
        self.setObjectName("libraryManagerDialog")
        self.image_db = image_db
        self.setWindowTitle("图像库管理")
        self.resize(1000, 700)
        
        self.current_page = 0
        self.page_size = 50
        self.loader_thread = None
        
        self._setup_ui()
        self.refresh_library()
        
    def _setup_ui(self):
        """设置UI布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("libraryManagerHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(14)

        title_copy = QVBoxLayout()
        title_copy.setContentsMargins(0, 0, 0, 0)
        title_copy.setSpacing(2)
        self.manager_title = QLabel("图像库管理")
        self.manager_title.setObjectName("libraryManagerTitle")
        self.manager_subtitle = QLabel("审阅素材、定位文件、批量清理与维护语义索引资产。")
        self.manager_subtitle.setObjectName("libraryManagerSubtitle")
        title_copy.addWidget(self.manager_title)
        title_copy.addWidget(self.manager_subtitle)
        header_layout.addLayout(title_copy, 1)

        self.total_value_label = self._create_header_metric(header_layout, "总素材", "--")
        self.loaded_value_label = self._create_header_metric(header_layout, "本页", "0 张")
        self.selected_value_label = self._create_header_metric(header_layout, "已选", "0 张")
        layout.addWidget(header)
        
        # 1. 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        
        self.btn_import = QPushButton("导入图片")
        self.btn_import.setProperty("variant", "primary")
        self.btn_import.setIcon(QIcon.fromTheme("document-new"))
        self.btn_import.clicked.connect(self._on_import_clicked)
        toolbar.addWidget(self.btn_import)
        
        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.setProperty("variant", "danger")
        self.btn_delete.setIcon(QIcon.fromTheme("edit-delete"))
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        toolbar.addWidget(self.btn_delete)
        
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setProperty("variant", "secondary")
        self.btn_refresh.setIcon(QIcon.fromTheme("view-refresh"))
        self.btn_refresh.clicked.connect(self.refresh_library)
        toolbar.addWidget(self.btn_refresh)
        
        toolbar.addStretch()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索图片 (标签/路径)...")
        self.search_input.setFixedWidth(250)
        self.search_input.returnPressed.connect(self._on_search)
        toolbar.addWidget(self.search_input)
        
        self.btn_search = QPushButton("搜索")
        self.btn_search.setProperty("variant", "secondary")
        self.btn_search.clicked.connect(self._on_search)
        toolbar.addWidget(self.btn_search)
        
        layout.addLayout(toolbar)
        
        # 2. 主要内容区 (Splitter: 列表 | 详情)
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧列表
        list_container = QWidget()
        list_container.setObjectName("libraryManagerListPane")
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(8)
        
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("libraryManagerList")
        self.list_widget.setIconSize(QSize(120, 120))
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        list_layout.addWidget(self.list_widget)
        
        # 分页控件
        pagination_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一页")
        self.btn_prev.setProperty("variant", "secondary")
        self.btn_prev.clicked.connect(self._prev_page)
        self.btn_next = QPushButton("下一页")
        self.btn_next.setProperty("variant", "secondary")
        self.btn_next.clicked.connect(self._next_page)
        self.lbl_page = QLabel("第 1 页")
        self.lbl_page.setObjectName("mutedText")
        
        pagination_layout.addWidget(self.btn_prev)
        pagination_layout.addWidget(self.lbl_page)
        pagination_layout.addWidget(self.btn_next)
        pagination_layout.addStretch()
        list_layout.addLayout(pagination_layout)
        
        splitter.addWidget(list_container)
        
        # 右侧详情
        self.detail_panel = QGroupBox("图片详情")
        self.detail_panel.setObjectName("libraryManagerDetailPanel")
        self.detail_panel.setMinimumWidth(260)
        detail_layout = QVBoxLayout(self.detail_panel)
        
        self.img_preview = QLabel("无预览")
        self.img_preview.setObjectName("libraryManagerPreview")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setMinimumHeight(200)
        detail_layout.addWidget(self.img_preview)
        
        form_layout = QFormLayout()
        self.lbl_filename = QLabel("-")
        self.lbl_resolution = QLabel("-")
        self.lbl_path = QLabel("-")
        self.lbl_path.setWordWrap(True)
        self.lbl_id = QLabel("-")
        
        form_layout.addRow("文件名:", self.lbl_filename)
        form_layout.addRow("分辨率:", self.lbl_resolution)
        form_layout.addRow("ID:", self.lbl_id)
        form_layout.addRow("路径:", self.lbl_path)
        
        detail_layout.addLayout(form_layout)
        detail_layout.addStretch()
        
        splitter.addWidget(self.detail_panel)
        splitter.setSizes([700, 300]) # 默认比例
        
        layout.addWidget(splitter)
        
        # 3. 状态栏
        self.status_bar = QLabel("就绪")
        self.status_bar.setObjectName("mutedText")
        layout.addWidget(self.status_bar)

        self._reset_detail_panel()
        self._update_selection_actions()

    def _create_header_metric(self, parent_layout: QHBoxLayout, name: str, value: str) -> QLabel:
        metric = QFrame()
        metric.setObjectName("libraryManagerMetric")
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        metric_layout.setSpacing(2)

        name_label = QLabel(name)
        name_label.setObjectName("libraryManagerMetricName")
        value_label = QLabel(value)
        value_label.setObjectName("libraryManagerMetricValue")

        metric_layout.addWidget(name_label)
        metric_layout.addWidget(value_label)
        parent_layout.addWidget(metric)
        return value_label

    def _sync_header_metrics(self, total_count=None, loaded_count=None):
        """同步管理弹窗顶部摘要。"""
        if not hasattr(self, "total_value_label"):
            return

        if total_count is None and self.image_db is not None:
            try:
                total_count = self.image_db.get_image_count()
            except Exception:
                total_count = None

        if loaded_count is None:
            loaded_count = len(getattr(self, "images_data", []) or [])

        selected_count = len(self.list_widget.selectedItems()) if hasattr(self, "list_widget") else 0
        self.total_value_label.setText(f"{total_count} 张" if total_count is not None else "--")
        self.loaded_value_label.setText(f"{loaded_count} 张")
        self.selected_value_label.setText(f"{selected_count} 张")

    def _reset_detail_panel(self):
        """清空详情区，避免列表刷新或取消选择后保留旧图片信息。"""
        self.lbl_filename.setText("-")
        self.lbl_resolution.setText("-")
        self.lbl_id.setText("-")
        self.lbl_path.setText("-")
        self.img_preview.clear()
        self.img_preview.setText("无预览")

    def _update_selection_actions(self):
        """同步依赖选中项的工具栏动作。"""
        self.btn_delete.setEnabled(bool(self.list_widget.selectedItems()))

    def _stop_thumbnail_loader(self, timeout_ms=None):
        """停止缩略图加载线程；关闭窗口时可设置超时。"""
        if not self.loader_thread or not self.loader_thread.isRunning():
            return True

        logger.debug("Stopping library thumbnail loader")
        self.loader_thread.stop()
        if timeout_ms is None:
            self.loader_thread.wait()
            return True

        stopped = self.loader_thread.wait(timeout_ms)
        if not stopped:
            logger.warning("Timed out while stopping library thumbnail loader")
        return stopped
        
    def refresh_library(self):
        """刷新列表"""
        self._load_page(self.current_page)
        
    def _load_page(self, page_index: int):
        """加载指定页"""
        if not self.image_db:
            self.images_data = []
            self.list_widget.clear()
            self._reset_detail_panel()
            self._update_selection_actions()
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.lbl_page.setText("无图像库")
            self.status_bar.setText("图像库未初始化")
            self._sync_header_metrics(None, 0)
            return
            
        self.list_widget.clear()
        self._reset_detail_panel()
        self._update_selection_actions()
        
        # 停止之前的加载线程
        self._stop_thumbnail_loader()
            
        offset = page_index * self.page_size
        
        # 判断是全部列表还是搜索结果（这里简化逻辑，暂只支持全部）
        # 如果需要支持搜索分页，需要修改ImageIndexDatabase的搜索接口支持分页
        # 目前搜索结果通常较少，可以一次性显示
        query = self.search_input.text().strip()
        total_count = None
        try:
            if query:
                # 搜索模式 (复用search_by_text，不支持分页)
                results = self.image_db.search_by_text(query, top_k=100)
                total_count = self.image_db.get_image_count()
                self.images_data = results
                self.current_page = 0
                self.btn_prev.setEnabled(False)
                self.btn_next.setEnabled(False)
                self.lbl_page.setText(f"搜索结果: {len(results)} 张")
            else:
                # 浏览模式
                total_count = self.image_db.get_image_count()
                self.images_data = self.image_db.get_all_images(limit=self.page_size, offset=offset)

                # 更新分页按钮状态
                self.btn_prev.setEnabled(page_index > 0)
                self.btn_next.setEnabled((offset + self.page_size) < total_count)
                self.lbl_page.setText(f"第 {page_index + 1} 页 (共 {total_count} 张)")
        except Exception as exc:
            self.images_data = []
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.lbl_page.setText("加载失败")
            self.status_bar.setText(f"加载失败: {exc}")
            self._sync_header_metrics(None, 0)
            return
            
        # 填充列表项
        for img in self.images_data:
            item = QListWidgetItem()
            path = img['path']
            name = Path(path).name
            item.setText(name)
            item.setData(Qt.UserRole, img) # 存储完整数据
            item.setToolTip(path)
            self.list_widget.addItem(item)
            
        if self.images_data:
            # 启动后台加载缩略图
            self.loader_thread = ThumbnailLoader(self.images_data)
            self.loader_thread.thumbnail_loaded.connect(self._update_item_icon)
            self.loader_thread.start()
        else:
            self.loader_thread = None
        
        self._sync_header_metrics(total_count, len(self.images_data))
        if query and not self.images_data:
            self.status_bar.setText(f"未找到匹配 \"{query}\" 的图片")
        else:
            self.status_bar.setText(f"已加载 {len(self.images_data)} 张图片")
        
    def _update_item_icon(self, img_id, q_image):
        """更新列表项图标"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole) or {}
            if data.get('id') == img_id:
                item.setIcon(QIcon(QPixmap.fromImage(q_image)))
                break
                
    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_library()
            
    def _next_page(self):
        self.current_page += 1
        self.refresh_library()
        
    def _on_search(self):
        self.current_page = 0
        self.refresh_library()
        
    def _on_selection_changed(self):
        """选中项改变"""
        self._update_selection_actions()
        self._sync_header_metrics()
        items = self.list_widget.selectedItems()
        if not items:
            self._reset_detail_panel()
            return
            
        # 显示第一个选中项的详情
        item = items[0]
        data = item.data(Qt.UserRole) or {}
        path = data.get('path', '')
        
        self.lbl_filename.setText(Path(path).name)
        self.lbl_id.setText(str(data.get('id', ''))[:8] + "...")
        self.lbl_path.setText(path)
        
        # 加载预览图
        try:
            image = imread_safe(path)
            if image is not None:
                h, w = image.shape[:2]
                self.lbl_resolution.setText(f"{w} x {h}")
                
                # 显示预览
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimg)
                
                # 适应Label大小
                scaled = pixmap.scaled(self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_preview.setPixmap(scaled)
            else:
                self.lbl_resolution.setText("-")
                self.img_preview.clear()
                self.img_preview.setText("加载失败")
        except Exception as exc:
            logger.debug("Failed to preview library image %s: %s", path, exc)
            self.lbl_resolution.setText("-")
            self.img_preview.clear()
            self.img_preview.setText("预览出错")

    def _on_import_clicked(self):
        """导入图片"""
        # 使用统一的图像选择对话框
        files = pick_images(self.image_db, multi_select=True, parent=self)

        if files:
            self.import_requested.emit(files)
            # 这里的导入是在主窗口处理的，可能需要一点时间
            # 可以暂时在状态栏提示
            self.status_bar.setText("正在后台导入...")
            
    def _on_delete_clicked(self):
        """删除选中"""
        items = self.list_widget.selectedItems()
        if not items:
            self.status_bar.setText("请先选择要删除的图片")
            self._update_selection_actions()
            return
            
        if QMessageBox.question(self, "确认删除", f"确定要从库中删除选中的 {len(items)} 张图片吗？") != QMessageBox.Yes:
            return
            
        removed = 0
        failures = []
        for item in items:
            data = item.data(Qt.UserRole) or {}
            img_id = data.get('id')
            if not img_id:
                failures.append(item.text())
                continue

            try:
                # 从数据库删除，不删除磁盘原文件
                self.image_db.remove_image(img_id)
                removed += 1
            except Exception as exc:
                label = Path(data.get('path', item.text())).name
                failures.append(f"{label}: {exc}")
                logger.debug("Failed to remove image %s from library: %s", img_id, exc)

        if removed:
            self.refresh_library()
            self.status_bar.setText(f"已从库中删除 {removed} 张图片")
        else:
            self._update_selection_actions()

        if failures:
            preview = "\n".join(failures[:3])
            if len(failures) > 3:
                preview += f"\n...另有 {len(failures) - 3} 张失败"
            QMessageBox.warning(self, "删除未完成", f"以下图片未能从库中删除:\n{preview}")
        elif removed:
            QMessageBox.information(self, "完成", f"已从库中删除 {removed} 张图片")

    def _show_context_menu(self, pos):
        """右键菜单"""
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        self._select_context_item(item)
            
        menu = QMenu(self)
        act_open = menu.addAction("打开文件位置")
        act_delete = menu.addAction("从库中删除")
        
        action = menu.exec_(self.list_widget.mapToGlobal(pos))
        
        if action == act_open:
            data = item.data(Qt.UserRole) or {}
            self._open_file_in_explorer(data.get('path', ''))
        elif action == act_delete:
            self._on_delete_clicked()

    def _select_context_item(self, item):
        """右键操作前选中目标项，避免删除仍作用于旧选择。"""
        if not item.isSelected():
            self.list_widget.clearSelection()
            item.setSelected(True)
            self.list_widget.setCurrentItem(item)
            
    def _open_file_in_explorer(self, path):
        directory = Path(path).expanduser().parent
        if not path or not directory.exists():
            self.status_bar.setText("无法打开文件位置：目录不存在")
            QMessageBox.warning(self, "无法打开文件位置", f"目录不存在:\n{directory}")
            return False

        try:
            if platform.system() == "Windows":
                os.startfile(str(directory))
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to open image directory %s", directory)
            self.status_bar.setText("无法打开文件位置")
            QMessageBox.warning(self, "无法打开文件位置", f"无法打开目录:\n{directory}\n\n{exc}")
            return False

        self.status_bar.setText(f"已打开文件位置: {directory}")
        return True
    
    def closeEvent(self, event):
        """关闭事件 - 确保线程正确停止"""
        if not self._stop_thumbnail_loader(3000):
            self.status_bar.setText("缩略图仍在加载中，暂不能关闭")
            event.ignore()
            return
        event.accept()

