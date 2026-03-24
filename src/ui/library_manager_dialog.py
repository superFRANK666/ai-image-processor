"""
图像库管理对话框
用于管理、浏览、删除和导入图像库资源
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import sys

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe

from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit,
    QMessageBox, QFileDialog, QSplitter, QGroupBox, QFormLayout,
    QProgressBar, QMenu
)
from PySide6.QtCore import Qt, Signal, QSize, QThread
from PySide6.QtGui import QIcon, QPixmap, QImage, QAction, QCursor

# 使用相对导入项目模块
from ..ai import ImageIndexDatabase
from ..core.config import SUPPORTED_FORMATS
from .image_picker_dialog import pick_images

class ThumbnailLoader(QThread):
    """后台加载缩略图线程"""
    thumbnail_loaded = Signal(str, QIcon)  # id, icon

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
                    scale = self.icon_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                    
                    # 转 RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(q_image)
                    
                    self.thumbnail_loaded.emit(img_id, QIcon(pixmap))
            except Exception:
                pass
                
    def stop(self):
        self._is_running = False

class LibraryManagerDialog(QDialog):
    """图像库管理器"""
    
    import_requested = Signal(list)  # 请求导入
    delete_requested = Signal(str)   # 请求删除
    
    def __init__(self, image_db: ImageIndexDatabase, parent=None):
        super().__init__(parent)
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
        
        # 1. 顶部工具栏
        toolbar = QHBoxLayout()
        
        self.btn_import = QPushButton("导入图片")
        self.btn_import.setIcon(QIcon.fromTheme("document-new"))
        self.btn_import.clicked.connect(self._on_import_clicked)
        toolbar.addWidget(self.btn_import)
        
        self.btn_delete = QPushButton("删除选中")
        self.btn_delete.setIcon(QIcon.fromTheme("edit-delete"))
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        toolbar.addWidget(self.btn_delete)
        
        self.btn_refresh = QPushButton("刷新")
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
        self.btn_search.clicked.connect(self._on_search)
        toolbar.addWidget(self.btn_search)
        
        layout.addLayout(toolbar)
        
        # 2. 主要内容区 (Splitter: 列表 | 详情)
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧列表
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0,0,0,0)
        
        self.list_widget = QListWidget()
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
        self.btn_prev.clicked.connect(self._prev_page)
        self.btn_next = QPushButton("下一页")
        self.btn_next.clicked.connect(self._next_page)
        self.lbl_page = QLabel("第 1 页")
        
        pagination_layout.addWidget(self.btn_prev)
        pagination_layout.addWidget(self.lbl_page)
        pagination_layout.addWidget(self.btn_next)
        pagination_layout.addStretch()
        list_layout.addLayout(pagination_layout)
        
        splitter.addWidget(list_container)
        
        # 右侧详情
        self.detail_panel = QGroupBox("图片详情")
        self.detail_panel.setWidth(300)
        detail_layout = QVBoxLayout(self.detail_panel)
        
        self.img_preview = QLabel("无预览")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setMinimumHeight(200)
        self.img_preview.setStyleSheet("background: #2d2d2d; border: 1px solid #444;")
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
        layout.addWidget(self.status_bar)
        
    def refresh_library(self):
        """刷新列表"""
        self._load_page(self.current_page)
        
    def _load_page(self, page_index: int):
        """加载指定页"""
        if not self.image_db:
            return
            
        self.list_widget.clear()
        self.lbl_filename.setText("-")
        self.img_preview.setText("无预览")
        
        # 停止之前的加载线程
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()
            self.loader_thread.wait()
            
        offset = page_index * self.page_size
        
        # 判断是全部列表还是搜索结果（这里简化逻辑，暂只支持全部）
        # 如果需要支持搜索分页，需要修改ImageIndexDatabase的搜索接口支持分页
        # 目前搜索结果通常较少，可以一次性显示
        if self.search_input.text().strip():
            # 搜索模式 (复用search_by_text，不支持分页)
            results = self.image_db.search_by_text(self.search_input.text().strip(), top_k=100)
            self.images_data = results
            self.current_page = 0
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.lbl_page.setText("搜索结果")
        else:
            # 浏览模式
            # 注意：get_all_images是从db获取
            total_count = self.image_db.get_image_count()
            self.images_data = self.image_db.get_all_images(limit=self.page_size, offset=offset)
            
            # 更新分页按钮状态
            self.btn_prev.setEnabled(page_index > 0)
            self.btn_next.setEnabled((offset + self.page_size) < total_count)
            self.lbl_page.setText(f"第 {page_index + 1} 页 (共 {total_count} 张)")
            
        # 填充列表项
        for img in self.images_data:
            item = QListWidgetItem()
            path = img['path']
            name = Path(path).name
            item.setText(name)
            item.setData(Qt.UserRole, img) # 存储完整数据
            item.setToolTip(path)
            self.list_widget.addItem(item)
            
        # 启动后台加载缩略图
        self.loader_thread = ThumbnailLoader(self.images_data)
        self.loader_thread.thumbnail_loaded.connect(self._update_item_icon)
        self.loader_thread.start()
        
        self.status_bar.setText(f"已加载 {len(self.images_data)} 张图片")
        
    def _update_item_icon(self, img_id, icon):
        """更新列表项图标"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data['id'] == img_id:
                item.setIcon(icon)
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
        items = self.list_widget.selectedItems()
        if not items:
            return
            
        # 显示第一个选中项的详情
        item = items[0]
        data = item.data(Qt.UserRole)
        path = data['path']
        
        self.lbl_filename.setText(Path(path).name)
        self.lbl_id.setText(str(data['id'])[:8] + "...")
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
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                # 适应Label大小
                scaled = pixmap.scaled(self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_preview.setPixmap(scaled)
            else:
                self.img_preview.setText("加载失败")
        except Exception:
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
            return
            
        if QMessageBox.question(self, "确认删除", f"确定要从库中删除选中的 {len(items)} 张图片吗？") != QMessageBox.Yes:
            return
            
        ids_to_remove = []
        for item in items:
            data = item.data(Qt.UserRole)
            img_id = data['id']
            # 从数据库删除
            self.image_db.remove_image(img_id)
            ids_to_remove.append(img_id)
            
        # 刷新页面
        self.refresh_library()
        QMessageBox.information(self, "完成", "删除成功")

    def _show_context_menu(self, pos):
        """右键菜单"""
        item = self.list_widget.itemAt(pos)
        if not item:
            return
            
        menu = QMenu()
        act_open = menu.addAction("打开文件位置")
        act_delete = menu.addAction("从库中删除")
        
        action = menu.exec_(self.list_widget.mapToGlobal(pos))
        
        if action == act_open:
            data = item.data(Qt.UserRole)
            self._open_file_in_explorer(data['path'])
        elif action == act_delete:
            self._on_delete_clicked()
            
    def _open_file_in_explorer(self, path):
        import platform, subprocess
        path = str(Path(path).parent)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    
    def closeEvent(self, event):
        """关闭事件 - 确保线程正确停止"""
        if self.loader_thread and self.loader_thread.isRunning():
            print("[LibraryManagerDialog] 停止缩略图加载线程...")
            self.loader_thread.stop()
            if not self.loader_thread.wait(3000):
                print("[LibraryManagerDialog] 警告: 线程等待超时，强制终止")
                self.loader_thread.terminate()
                self.loader_thread.wait()
        event.accept()

