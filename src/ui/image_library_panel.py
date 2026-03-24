"""
图像库面板
显示已索引的图像和搜索结果
"""
import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import cv2

# 导入中文路径安全的图像IO函数 (使用相对导入)
from ..utils.image_io import imread as imread_safe

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QLineEdit, QMenu, QComboBox, QInputDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QCursor

from .image_picker_dialog import pick_images

# 类型检查时导入（不影响运行时）
if TYPE_CHECKING:
    from ..ai import ImageIndexDatabase


class ImageThumbnailWidget(QWidget):
    """图像缩略图组件 (包含图片和文字)"""

    clicked = Signal(str)  # 图像路径
    double_clicked = Signal(str)
    rename_requested = Signal(str, str, str) # id, old_name, new_name
    delete_requested = Signal(str) # id

    def __init__(self, image_path: str, size: int = 120, name: str = "", img_id: str = ""):
        super().__init__()
        self.image_path = image_path
        self.img_name = name
        self.img_id = img_id
        
        self.setFixedWidth(size + 10)
        self._setup_ui(size)

    def _setup_ui(self, size):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # 图片容器
        self.img_label = QLabel()
        self.img_label.setFixedSize(size, size)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 4px;
            }
        """)
        self.img_label.setCursor(Qt.PointingHandCursor)
        self._load_thumbnail(size)
        layout.addWidget(self.img_label)
        
        # 文字标签
        self.name_label = QLabel(self.img_name)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        
        # 省略长文本
        font_metrics = self.name_label.fontMetrics()
        elided_text = font_metrics.elidedText(self.img_name, Qt.ElideMiddle, size)
        self.name_label.setText(elided_text)
        self.name_label.setToolTip(self.img_name) # 悬停显示全名
        
        layout.addWidget(self.name_label)
        
    def _load_thumbnail(self, size: int):
        try:
            image = imread_safe(self.image_path)
            if image is not None:
                # 保持比例缩放并裁剪/填充到正方形
                h, w = image.shape[:2]
                scale = size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_data = np.ascontiguousarray(rgb).copy()
                q_image = QImage(img_data.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
                self.img_label.setPixmap(QPixmap.fromImage(q_image.copy()))
            else:
                self.img_label.setText("Err")
        except Exception:
            self.img_label.setText("Err")
            
    def set_selected(self, selected: bool):
        if selected:
            self.img_label.setStyleSheet("QLabel { background: #404040; border: 2px solid #0078d4; border-radius: 4px; }")
            self.name_label.setStyleSheet("color: #0078d4; font-weight: bold; font-size: 11px;")
        else:
            self.img_label.setStyleSheet("QLabel { background: #2d2d2d; border: 2px solid #404040; border-radius: 4px; }")
            self.name_label.setStyleSheet("color: #cccccc; font-size: 11px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)
            
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 判断点击位置，如果是点文字则重命名，点图片则打开
            # 简化逻辑：点哪里双击都触发打开，重命名通过专门的动作或者名字双击（有点复杂，不如双击名字重命名）
            child = self.childAt(event.pos())
            if child == self.name_label:
                self._start_rename()
            else:
                self.double_clicked.emit(self.image_path)
                
    def contextMenuEvent(self, event):
        """右键菜单"""
        menu = QMenu(self)
        
        rename_action = menu.addAction("重命名")
        rename_action.triggered.connect(self._start_rename)
        
        delete_action = menu.addAction("删除")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(self.img_id))
        
        menu.addSeparator()
        
        open_action = menu.addAction("打开")
        open_action.triggered.connect(lambda: self.double_clicked.emit(self.image_path))
        
        menu.exec_(QCursor.pos())

    def _start_rename(self):
        new_name, ok = QInputDialog.getText(self, "重命名", "输入新名称:", text=self.img_name)
        if ok and new_name and new_name != self.img_name:
            self.rename_requested.emit(self.img_id, self.img_name, new_name)
            self.img_name = new_name
            font_metrics = self.name_label.fontMetrics()
            elided_text = font_metrics.elidedText(self.img_name, Qt.ElideMiddle, self.img_label.width())
            self.name_label.setText(elided_text)
            self.name_label.setToolTip(self.img_name)


class ImageLibraryPanel(QWidget):
    """图像库面板"""

    image_selected = Signal(str)  # 选中的图像路径
    import_requested = Signal(list, str)  # 请求导入图像列表(files, group)

    def __init__(self, image_db: Optional["ImageIndexDatabase"] = None):
        super().__init__()
        self.image_db = image_db
        self._thumbnails: List[ImageThumbnailWidget] = []
        self._setup_ui()

    def set_database(self, image_db: "ImageIndexDatabase"):
        """设置数据库实例"""
        self.image_db = image_db
        self.refresh()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 搜索栏
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索图像...")
        self.search_input.returnPressed.connect(self._on_search)
        search_layout.addWidget(self.search_input)

        # 分组筛选
        self.group_combo = QComboBox()
        self.group_combo.addItem("全部")
        self.group_combo.addItem("默认")
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.group_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_combo.customContextMenuRequested.connect(self._on_group_context_menu)
        search_layout.addWidget(self.group_combo)

        # 新建分组
        self.new_group_btn = QPushButton()
        self.new_group_btn.setText("+")
        self.new_group_btn.setFixedSize(28, 28)
        self.new_group_btn.setToolTip("新建分组")
        # 使用样式表美化 "+" 号
        self.new_group_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 18px;
                font-family: Arial, sans-serif;
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #e0e0e0;
                padding: 0px;
                padding-bottom: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: #ffffff;
                border-color: #0078d4;
            }
        """)
        self.new_group_btn.clicked.connect(self._on_new_group)
        search_layout.addWidget(self.new_group_btn)
        
        # 增加重命名图片按钮 (移除，改为双击名称)
        # self.rename_btn = QPushButton("命名")
        # self.rename_btn.setToolTip("重命名选中图片的显示名称/标签")
        # self.rename_btn.clicked.connect(self._on_rename_clicked)
        # self.rename_btn.setEnabled(False) # 初始禁用
        # search_layout.addWidget(self.rename_btn)


        self.import_btn = QPushButton("导入")
        self.import_btn.setToolTip("导入图片到库")
        self.import_btn.clicked.connect(self._on_import_btn_clicked)
        search_layout.addWidget(self.import_btn)

        self.search_btn = QPushButton("搜索")
        self.search_btn.clicked.connect(self._on_search)
        search_layout.addWidget(self.search_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        search_layout.addWidget(self.refresh_btn)

        # 重建索引按钮
        self.rebuild_btn = QPushButton("重建索引")
        self.rebuild_btn.setToolTip("重建所有图片的语义索引（用于改进搜索）")
        self.rebuild_btn.clicked.connect(self._on_rebuild_index)
        search_layout.addWidget(self.rebuild_btn)

        layout.addLayout(search_layout)

        # 状态标签
        count = 0
        if self.image_db:
            try:
                count = self.image_db.get_image_count()
            except Exception:
                pass
        self.status_label = QLabel(f"图像库: {count} 张图片")
        layout.addWidget(self.status_label)

        # 缩略图滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: #1a1a1a; }")

        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setSpacing(5)
        self.thumbnail_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.thumbnail_widget)
        layout.addWidget(scroll_area)

    def refresh(self):
        """刷新图像列表"""
        # 更新分组
        current_group = self.group_combo.currentText()
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("全部")
        if self.image_db:
             groups = self.image_db.get_groups()
             self.group_combo.addItems(groups)
        
        index = self.group_combo.findText(current_group)
        if index >= 0:
            self.group_combo.setCurrentIndex(index)
        else:
            self.group_combo.setCurrentIndex(0)
        self.group_combo.blockSignals(False)

        current_filter = self.group_combo.currentText()
        if current_filter == "全部":
             self._load_images("全部")
        else:
             self._load_images(current_filter)

    def _load_images(self, group: str):
        """加载特定分组图像"""
        self._clear_thumbnails()
        if not self.image_db:
            return
            
        images = self.image_db.get_images_by_group(group, limit=50) # 简单取前50张
        self.show_search_results(images)
        self.status_bar_update(len(images))

    def status_bar_update(self, count):
         try:
             if self.image_db is None:
                 self.status_label.setText(f"当前显示: {count}")
                 return
             total = self.image_db.get_image_count()
             self.status_label.setText(f"当前显示: {count} / 总计: {total}")
         except Exception:
             self.status_label.setText(f"当前显示: {count}")

    def _clear_thumbnails(self):
        """清除所有缩略图"""
        self.selected_thumb = None
        # self.rename_btn.setEnabled(False) # 移除
        for thumb in self._thumbnails:
            thumb.deleteLater()
        self._thumbnails.clear()

        # 清除布局中的所有项
        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_search_results(self, results: List[Dict[str, Any]]):
        """显示搜索结果"""
        self._clear_thumbnails()

        if not results:
            self.status_label.setText("未找到匹配的图片")
            return

        self.status_label.setText(f"找到 {len(results)} 张相似图片")

        # 计算每行显示的数量
        panel_width = self.width() - 30
        thumb_size = 120
        cols = max(1, panel_width // (thumb_size + 5))

        for i, result in enumerate(results):
            image_path = result.get("path", "")
            if not image_path or not Path(image_path).exists():
                continue

            # 改用 ImageThumbnailWidget
            name = result.get("metadata", {}).get("name", Path(image_path).name)
            img_id = result.get("id")
            
            thumb = ImageThumbnailWidget(image_path, thumb_size, name, img_id)
            thumb.clicked.connect(self._on_thumbnail_clicked)
            thumb.double_clicked.connect(self._on_thumbnail_double_clicked)
            thumb.rename_requested.connect(self._on_rename_requested_from_thumb)
            thumb.delete_requested.connect(self._on_delete_requested_from_thumb)

            row = i // cols
            col = i % cols
            self.thumbnail_layout.addWidget(thumb, row, col)
            self._thumbnails.append(thumb)

            # 添加相似度标签
            similarity = result.get("similarity", 0)
            if similarity > 0:
                thumb.setToolTip(f"{name}\n相似度: {similarity:.2%}\n{image_path}")

            # thumb.setToolTip(f"{name}\n相似度: {similarity:.2%}\n{image_path}") # 已经在上面设置了
            
            # 可以在缩略图下方显示名字? 为了简洁暂时不做，悬停显示即可
            # 或者给ImageThumbnail增加显示名字的能力
            # thumb.data_id = result.get("id") # 存储ID方便操作
            # thumb.data_name = name

    def _on_search(self):
        """执行搜索"""
        query = self.search_input.text().strip()
        if not query:
            # 空查询时刷新显示全部
            self.refresh()
            return

        if not self.image_db:
            self.status_label.setText("图像库未初始化")
            return

        results = self.image_db.search_by_text(query, top_k=20)

        if not results:
            self.status_label.setText(f"未找到匹配 \"{query}\" 的图片")
            self._clear_thumbnails()
            return

        self.show_search_results(results)

    def _on_thumbnail_clicked(self, image_path: str):
        """缩略图点击"""
        # 查找对应的thumb对象
        self.selected_thumb = None
        for thumb in self._thumbnails:
            if thumb.image_path == image_path:
                self.selected_thumb = thumb
                thumb.set_selected(True)
            else:
                thumb.set_selected(False)
        
        # 发送选中信号，以便主窗口可以在左侧显示
        self.image_selected.emit(image_path)
        
        # self.rename_btn.setEnabled(True) # 移除

    def _on_rename_requested_from_thumb(self, img_id, old_name, new_name):
        """缩略图请求重命名"""
        if self.image_db:
             self.image_db.update_image_metadata(img_id, {"name": new_name})
             # 这里不需要刷新整个列表，因为widget自己已经更新了文字
             self.status_label.setText(f"已重命名: {old_name} -> {new_name}")

    def _on_delete_requested_from_thumb(self, img_id):
        """缩略图请求删除"""
        self._delete_image(img_id)
        
    def _delete_image(self, img_id):
        """删除图像逻辑"""
        from PySide6.QtWidgets import QMessageBox
        if QMessageBox.question(self, "确认删除", "确定要删除这张图片吗？") != QMessageBox.Yes:
            return
            
        if self.image_db:
            self.image_db.remove_image(img_id)
            self.refresh()
            self.status_label.setText("已删除图像")
            
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Delete:
            if hasattr(self, 'selected_thumb') and self.selected_thumb:
                 self._delete_image(self.selected_thumb.img_id)
            else:
                # 检查是否需要删除当前分组（如果是非默认分组且为空? 暂不实现复杂分组删除）
                pass
        super().keyPressEvent(event)

    def _on_thumbnail_double_clicked(self, image_path: str):
        """缩略图双击"""
        self.image_selected.emit(image_path)

    def resizeEvent(self, event):
        """窗口大小改变"""
        super().resizeEvent(event)
        # 可以重新排列缩略图

    def _on_group_changed(self, index):
        """分组改变"""
        # 防止递归刷新
        if self.group_combo.signalsBlocked():
            return
        self.refresh()

    def _on_rename_clicked(self):
        """重命名图片（更新元数据中的name）"""
        if not hasattr(self, 'selected_thumb') or not self.selected_thumb:
            return
            
        old_name = getattr(self.selected_thumb, 'data_name', Path(self.selected_thumb.image_path).name)
        img_id = getattr(self.selected_thumb, 'data_id', None)
        
        if not img_id:
            return

        new_name, ok = QInputDialog.getText(self, "重命名图片", "请输入新名称:", text=old_name)
        if ok and new_name:
             # 更新数据库
             if self.image_db:
                 self.image_db.update_image_metadata(img_id, {"name": new_name})
                 # 刷新界面
                 self.refresh()

    def _on_new_group(self):
        """新建分组"""
        name, ok = QInputDialog.getText(self, "新建分组", "请输入分组名称:")
        if ok and name:
             # 添加到下拉框
             self.group_combo.blockSignals(True)
             if self.group_combo.findText(name) == -1:
                 self.group_combo.addItem(name)
             self.group_combo.setCurrentText(name)
             self.group_combo.blockSignals(False)

             # 持久化保存分组到数据库
             if self.image_db:
                 self.image_db.add_group(name)

             # 刷新显示
             self._load_images(name)

    def _on_group_context_menu(self, pos):
        """分组下拉框右键菜单"""
        current_group = self.group_combo.currentText()

        # 不允许删除 "全部" 和 "默认" 分组
        if current_group in ["全部", "默认"]:
            return

        menu = QMenu(self)

        delete_action = menu.addAction(f"删除分组 '{current_group}'")
        delete_action.triggered.connect(lambda: self._delete_group(current_group))

        rename_action = menu.addAction(f"重命名分组 '{current_group}'")
        rename_action.triggered.connect(lambda: self._rename_group(current_group))

        menu.exec_(self.group_combo.mapToGlobal(pos))

    def _delete_group(self, group_name: str):
        """删除分组"""
        from PySide6.QtWidgets import QMessageBox

        # 确认删除
        result = QMessageBox.question(
            self, "确认删除",
            f"确定要删除分组 '{group_name}' 吗？\n分组内的图片将移动到 '默认' 分组。",
            QMessageBox.Yes | QMessageBox.No
        )

        if result != QMessageBox.Yes:
            return

        # 从数据库删除分组（移动图片到默认分组）
        if self.image_db:
            self.image_db.delete_group(group_name)

        # 从下拉框移除
        index = self.group_combo.findText(group_name)
        if index >= 0:
            self.group_combo.removeItem(index)

        # 切换到全部分组
        self.group_combo.setCurrentIndex(0)
        self.refresh()

        self.status_label.setText(f"已删除分组: {group_name}")

    def _rename_group(self, old_name: str):
        """重命名分组"""
        new_name, ok = QInputDialog.getText(self, "重命名分组", "请输入新名称:", text=old_name)
        if ok and new_name and new_name != old_name:
            # 更新数据库
            if self.image_db:
                self.image_db.rename_group(old_name, new_name)

            # 更新下拉框
            index = self.group_combo.findText(old_name)
            if index >= 0:
                self.group_combo.setItemText(index, new_name)

            self.refresh()
            self.status_label.setText(f"已重命名分组: {old_name} -> {new_name}")

    def _on_rebuild_index(self):
        """重建所有图片的语义索引"""
        from PySide6.QtWidgets import QMessageBox, QProgressDialog
        from PySide6.QtCore import Qt

        if not self.image_db:
            QMessageBox.warning(self, "错误", "图像库未初始化")
            return

        # 确认
        result = QMessageBox.question(
            self, "重建索引",
            "将使用AI模型重新分析所有图片，以提高语义搜索准确度。\n\n"
            "这个过程可能需要几分钟，是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )

        if result != QMessageBox.Yes:
            return

        # 禁用按钮
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setText("重建中...")

        try:
            # 创建进度对话框
            count = self.image_db.get_image_count()
            progress = QProgressDialog("正在重建索引...", "取消", 0, count, self)
            progress.setWindowTitle("重建索引")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            def update_progress(current, total):
                progress.setValue(current)
                progress.setLabelText(f"正在处理: {current}/{total}")
                from PySide6.QtWidgets import QApplication
                QApplication.processEvents()
                if progress.wasCanceled():
                    raise InterruptedError("用户取消")

            # 执行重建
            rebuilt = self.image_db.rebuild_all_indexes(progress_callback=update_progress)

            progress.close()

            QMessageBox.information(
                self, "完成",
                f"索引重建完成！\n成功处理 {rebuilt} 张图片。\n\n"
                "现在可以使用中文语义搜索了，例如搜索\"杯子\"、\"风景\"等。"
            )

            self.refresh()

        except InterruptedError:
            self.status_label.setText("索引重建已取消")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重建索引失败: {e}")
        finally:
            self.rebuild_btn.setEnabled(True)
            self.rebuild_btn.setText("重建索引")

    def _on_import_btn_clicked(self):
        """导入按钮点击"""
        # 使用统一的图像选择对话框
        files = pick_images(self.image_db, multi_select=True, parent=self)

        if files:
            # 询问分组
            groups = [self.group_combo.itemText(i) for i in range(self.group_combo.count()) if self.group_combo.itemText(i) != "全部"]
            if not groups: groups = ["默认"]

            group, ok = QInputDialog.getItem(self, "选择分组", "请选择导入的分组:", groups, 0, True)
            if ok and group:
                self.import_requested.emit(files, group)

