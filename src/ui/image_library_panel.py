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
    QScrollArea, QGridLayout, QLineEdit, QMenu, QComboBox, QInputDialog,
    QMessageBox, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage, QCursor

from .image_picker_dialog import pick_images
from .ui_utils import fit_thumbnail_size

# 类型检查时导入（不影响运行时）
if TYPE_CHECKING:
    from ..ai import ImageIndexDatabase


RESERVED_GROUP_NAMES = {"全部", "默认"}


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
        self.img_label.setObjectName("thumbnailImage")
        self.img_label.setFixedSize(size, size)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setCursor(Qt.PointingHandCursor)
        self.img_label.setProperty("selected", False)
        self._load_thumbnail(size)
        layout.addWidget(self.img_label)
        
        # 文字标签
        self.name_label = QLabel(self.img_name)
        self.name_label.setObjectName("thumbnailName")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setProperty("selected", False)
        
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
                new_w, new_h = fit_thumbnail_size(w, h, size)
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
        self.img_label.setProperty("selected", selected)
        self.name_label.setProperty("selected", selected)
        for widget in (self.img_label, self.name_label):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

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
        if not ok:
            return

        normalized = str(new_name).strip()
        if normalized != self.img_name:
            self.rename_requested.emit(self.img_id, self.img_name, normalized)
        if normalized:
            self.img_name = normalized
            font_metrics = self.name_label.fontMetrics()
            elided_text = font_metrics.elidedText(self.img_name, Qt.ElideMiddle, self.img_label.width())
            self.name_label.setText(elided_text)
            self.name_label.setToolTip(self.img_name)


class RebuildIndexThread(QThread):
    """后台重建图像语义索引。"""

    progress = Signal(int, int)
    rebuild_finished = Signal(int)
    rebuild_failed = Signal(str)
    rebuild_canceled = Signal()

    def __init__(self, image_db):
        super().__init__()
        self.image_db = image_db
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            def update_progress(current, total):
                if self._stop_requested:
                    raise InterruptedError("用户取消")
                self.progress.emit(current, total)

            rebuilt = self.image_db.rebuild_all_indexes(progress_callback=update_progress)
            if self._stop_requested:
                self.rebuild_canceled.emit()
            else:
                self.rebuild_finished.emit(rebuilt)
        except InterruptedError:
            self.rebuild_canceled.emit()
        except Exception as e:
            self.rebuild_failed.emit(str(e))


class ImageLibraryPanel(QWidget):
    """图像库面板"""

    image_selected = Signal(str)  # 选中的图像路径
    import_requested = Signal(list, str)  # 请求导入图像列表(files, group)

    def __init__(self, image_db: Optional["ImageIndexDatabase"] = None):
        super().__init__()
        self.image_db = image_db
        self._rebuild_thread: Optional[RebuildIndexThread] = None
        self._rebuild_progress = None
        self._thumbnails: List[ImageThumbnailWidget] = []
        self.empty_state: Optional[QFrame] = None
        self.selected_thumb = None
        self._setup_ui()

    def set_database(self, image_db: "ImageIndexDatabase"):
        """设置数据库实例"""
        self.image_db = image_db
        self.refresh()

    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header_frame = QFrame()
        header_frame.setObjectName("libraryHeader")
        header = QVBoxLayout(header_frame)
        header.setContentsMargins(14, 12, 14, 12)
        header.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_copy = QVBoxLayout()
        title_copy.setContentsMargins(0, 0, 0, 0)
        title_copy.setSpacing(2)
        self.title_label = QLabel("素材库")
        self.title_label.setObjectName("libraryTitle")
        self.subtitle_label = QLabel("组织、检索和复用当前项目的视觉素材。")
        self.subtitle_label.setObjectName("librarySubtitle")
        self.subtitle_label.setWordWrap(True)
        title_copy.addWidget(self.title_label)
        title_copy.addWidget(self.subtitle_label)
        title_row.addLayout(title_copy)
        title_row.addStretch()
        self.library_state_badge = QLabel("准备中")
        self.library_state_badge.setObjectName("libraryStateBadge")
        title_row.addWidget(self.library_state_badge, 0, Qt.AlignTop)
        header.addLayout(title_row)

        summary_strip = QFrame()
        summary_strip.setObjectName("librarySummaryStrip")
        summary_layout = QHBoxLayout(summary_strip)
        summary_layout.setContentsMargins(10, 8, 10, 8)
        summary_layout.setSpacing(8)
        self.total_value_label = self._create_summary_metric(summary_layout, "总素材", "--")
        self.visible_value_label = self._create_summary_metric(summary_layout, "当前显示", "0 张")
        self.group_value_label = self._create_summary_metric(summary_layout, "分组", "全部")
        header.addWidget(summary_strip)
        layout.addWidget(header_frame)

        # 搜索与分组筛选
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("librarySearch")
        self.search_input.setPlaceholderText("搜索图像、风格或标签...")
        self.search_input.returnPressed.connect(self._on_search)
        filter_layout.addWidget(self.search_input, 1)

        # 分组筛选
        self.group_combo = QComboBox()
        self.group_combo.addItem("全部")
        self.group_combo.addItem("默认")
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        self.group_combo.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_combo.customContextMenuRequested.connect(self._on_group_context_menu)
        filter_layout.addWidget(self.group_combo)

        # 新建分组
        self.new_group_btn = QPushButton()
        self.new_group_btn.setText("+")
        self.new_group_btn.setFixedSize(28, 28)
        self.new_group_btn.setToolTip("新建分组")
        self.new_group_btn.setProperty("variant", "secondary")
        self.new_group_btn.clicked.connect(self._on_new_group)
        filter_layout.addWidget(self.new_group_btn)
        layout.addLayout(filter_layout)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        
        # 增加重命名图片按钮 (移除，改为双击名称)
        # self.rename_btn = QPushButton("命名")
        # self.rename_btn.setToolTip("重命名选中图片的显示名称/标签")
        # self.rename_btn.clicked.connect(self._on_rename_clicked)
        # self.rename_btn.setEnabled(False) # 初始禁用
        # search_layout.addWidget(self.rename_btn)


        self.import_btn = QPushButton("导入")
        self.import_btn.setProperty("variant", "primary")
        self.import_btn.setToolTip("导入图片到库")
        self.import_btn.clicked.connect(self._on_import_btn_clicked)
        action_layout.addWidget(self.import_btn)

        self.search_btn = QPushButton("搜索")
        self.search_btn.setProperty("variant", "secondary")
        self.search_btn.clicked.connect(self._on_search)
        action_layout.addWidget(self.search_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setProperty("variant", "secondary")
        self.refresh_btn.clicked.connect(self.refresh)
        action_layout.addWidget(self.refresh_btn)

        # 重建索引按钮
        self.rebuild_btn = QPushButton("重建索引")
        self.rebuild_btn.setProperty("variant", "secondary")
        self.rebuild_btn.setToolTip("重建所有图片的语义索引（用于改进搜索）")
        self.rebuild_btn.clicked.connect(self._on_rebuild_index)
        action_layout.addWidget(self.rebuild_btn)
        action_layout.addStretch()

        layout.addLayout(action_layout)

        # 状态标签
        count = 0
        if self.image_db:
            try:
                count = self.image_db.get_image_count()
            except Exception:
                pass
        self.status_label = QLabel(f"图像库: {count} 张图片")
        self.status_label.setObjectName("mutedText")
        layout.addWidget(self.status_label)

        self.summary_label = QLabel("等待图像库加载")
        self.summary_label.setObjectName("librarySummaryText")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        # 缩略图滚动区域
        scroll_area = QScrollArea()
        scroll_area.setObjectName("libraryScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.thumbnail_widget = QWidget()
        self.thumbnail_widget.setObjectName("thumbnailGrid")
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(8)
        self.thumbnail_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll_area.setWidget(self.thumbnail_widget)
        layout.addWidget(scroll_area)
        self._update_library_controls()

    def _create_summary_metric(self, parent_layout: QHBoxLayout, name: str, value: str) -> QLabel:
        metric = QFrame()
        metric.setObjectName("libraryMetric")
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        metric_layout.setSpacing(2)

        name_label = QLabel(name)
        name_label.setObjectName("libraryMetricName")
        value_label = QLabel(value)
        value_label.setObjectName("libraryMetricValue")
        value_label.setWordWrap(False)

        metric_layout.addWidget(name_label)
        metric_layout.addWidget(value_label)
        parent_layout.addWidget(metric, 1)
        return value_label

    def refresh(self):
        """刷新图像列表"""
        self._update_library_controls()
        # 更新分组
        current_group = self.group_combo.currentText()
        group_error = None
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("全部")
        if self.image_db:
            try:
                groups = self.image_db.get_groups()
                self.group_combo.addItems(groups)
            except Exception as e:
                group_error = f"加载分组失败: {e}"
        
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

        if group_error:
            self.status_label.setText(group_error)

    def _update_library_controls(self):
        """根据图像库是否可用同步检索和管理控件状态。"""
        available = self.image_db is not None
        for widget in (
                self.search_input,
                self.group_combo,
                self.new_group_btn,
                self.search_btn,
                self.refresh_btn,
                self.rebuild_btn):
            widget.setEnabled(available)

        self.search_input.setPlaceholderText(
            "搜索图像..." if available else "图像库加载后可搜索..."
        )
        if not available:
            self.status_label.setText("图像库未初始化")
            self._set_summary("素材库暂不可用。图像数据库加载完成后，可在这里检索和管理素材。")
            self._sync_summary_metrics(0)
            self._show_empty_state("图像库正在准备", "模型和索引加载完成后，素材会显示在这里。")
        else:
            self._sync_summary_metrics(len(self._thumbnails))

    def _load_images(self, group: str):
        """加载特定分组图像"""
        self._clear_thumbnails()
        if not self.image_db:
            self.status_label.setText("图像库未初始化")
            self._set_summary("素材库暂不可用。")
            self._show_empty_state("图像库未初始化", "请等待后台图像数据库加载完成。")
            return

        try:
            images = self.image_db.get_images_by_group(group, limit=50) # 简单取前50张
        except Exception as e:
            self.status_label.setText(f"加载图片失败: {e}")
            self._set_summary("无法读取当前分组。")
            self._show_empty_state("加载失败", str(e))
            return

        if not images:
            total = 0
            try:
                total = self.image_db.get_image_count()
            except Exception:
                pass
            self.status_label.setText(f"当前显示: 0 / 总计: {total}")
            self._sync_summary_metrics(0, total)
            self._set_summary(f"当前分组: {self.group_combo.currentText()} · 暂无素材")
            if total:
                self._show_empty_state("当前分组暂无素材", "切换分组或导入素材到当前分组后即可继续浏览。")
            else:
                self._show_empty_state("素材库为空", "导入参考图、风格样张或项目素材后即可开始检索。")
            return

        shown_count = self.show_search_results(images)
        self.status_bar_update(shown_count)

    def status_bar_update(self, count):
         try:
             if self.image_db is None:
                 self.status_label.setText(f"当前显示: {count}")
                 return
             total = self.image_db.get_image_count()
             self.status_label.setText(f"当前显示: {count} / 总计: {total}")
             self._sync_summary_metrics(count, total)
             self._set_summary(
                 f"当前分组: {self.group_combo.currentText()} · 显示 {count} 张 · 总计 {total} 张"
             )
         except Exception:
             self.status_label.setText(f"当前显示: {count}")
             self._sync_summary_metrics(count)
             self._set_summary(f"当前显示 {count} 张")

    def _clear_thumbnails(self):
        """清除所有缩略图"""
        self.selected_thumb = None
        self.empty_state = None
        # self.rename_btn.setEnabled(False) # 移除
        for thumb in self._thumbnails:
            thumb.deleteLater()
        self._thumbnails.clear()

        # 清除布局中的所有项
        while self.thumbnail_layout.count():
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_search_results(self, results: List[Dict[str, Any]]) -> int:
        """显示搜索结果"""
        self._clear_thumbnails()

        if not results:
            self.status_label.setText("未找到匹配的图片")
            self._sync_summary_metrics(0)
            self._set_summary("没有匹配结果")
            self._show_empty_state("没有匹配结果", "尝试更换关键词，或导入更多素材后再搜索。")
            return 0

        # 计算每行显示的数量
        panel_width = self.width() - 30
        thumb_size = 120
        cols = max(1, panel_width // (thumb_size + 5))

        displayed_count = 0
        for result in results:
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

            row = displayed_count // cols
            col = displayed_count % cols
            self.thumbnail_layout.addWidget(thumb, row, col)
            self._thumbnails.append(thumb)
            displayed_count += 1

            # 添加相似度标签
            similarity = result.get("similarity", 0)
            if similarity > 0:
                thumb.setToolTip(f"{name}\n相似度: {similarity:.2%}\n{image_path}")

            # thumb.setToolTip(f"{name}\n相似度: {similarity:.2%}\n{image_path}") # 已经在上面设置了
            
            # 可以在缩略图下方显示名字? 为了简洁暂时不做，悬停显示即可
            # 或者给ImageThumbnail增加显示名字的能力
            # thumb.data_id = result.get("id") # 存储ID方便操作
            # thumb.data_name = name
        if displayed_count:
            self.status_label.setText(f"找到 {displayed_count} 张相似图片")
            self._sync_summary_metrics(displayed_count)
            self._set_summary(f"结果已更新 · 显示 {displayed_count} 张")
        else:
            self.status_label.setText("未找到可显示的图片")
            self._sync_summary_metrics(0)
            self._set_summary("结果中没有可显示的本地文件")
            self._show_empty_state("没有可显示的素材", "结果中的文件可能已移动或被删除。")

        return displayed_count

    def _on_search(self):
        """执行搜索"""
        query = self.search_input.text().strip()
        if not query:
            # 空查询时刷新显示全部
            self.refresh()
            return

        if not self.image_db:
            self.status_label.setText("图像库未初始化")
            self._sync_summary_metrics(0)
            return

        try:
            results = self.image_db.search_by_text(query, top_k=20)
        except Exception as e:
            self.status_label.setText(f"搜索失败: {e}")
            return

        if not results:
            self.status_label.setText(f"未找到匹配 \"{query}\" 的图片")
            self._clear_thumbnails()
            self._sync_summary_metrics(0)
            self._set_summary(f"没有匹配关键词: {query}")
            self._show_empty_state("没有匹配结果", f"未找到与 \"{query}\" 相关的素材。")
            return

        self.show_search_results(results)

    def _set_summary(self, text: str):
        """更新素材库摘要。"""
        if hasattr(self, "summary_label"):
            self.summary_label.setText(text)

    def _sync_summary_metrics(self, visible_count: int, total_count: Optional[int] = None):
        """同步素材库顶部的关键状态指标。"""
        if not hasattr(self, "total_value_label"):
            return

        if self.image_db is None:
            self.total_value_label.setText("--")
            self.visible_value_label.setText(f"{visible_count} 张")
            self.group_value_label.setText("未连接")
            self.library_state_badge.setText("未连接")
            self.library_state_badge.setProperty("tone", "offline")
        else:
            total = total_count
            if total is None:
                try:
                    total = self.image_db.get_image_count()
                except Exception:
                    total = None

            self.total_value_label.setText(f"{total} 张" if total is not None else "--")
            self.visible_value_label.setText(f"{visible_count} 张")
            self.group_value_label.setText(self.group_combo.currentText() or "全部")
            self.library_state_badge.setText("就绪")
            self.library_state_badge.setProperty("tone", "ready")

        self.library_state_badge.style().unpolish(self.library_state_badge)
        self.library_state_badge.style().polish(self.library_state_badge)

    def _show_empty_state(self, title: str, subtitle: str):
        """在缩略图网格中显示空状态。"""
        if self.empty_state is not None:
            for index in range(self.thumbnail_layout.count()):
                item = self.thumbnail_layout.itemAt(index)
                if item and item.widget() is self.empty_state:
                    self.thumbnail_layout.takeAt(index)
                    break
            self.empty_state.deleteLater()
            self.empty_state = None

        self.empty_state = QFrame()
        self.empty_state.setObjectName("libraryEmptyState")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.setSpacing(10)

        self.empty_title_label = QLabel(title)
        self.empty_title_label.setObjectName("libraryEmptyTitle")
        self.empty_title_label.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.empty_title_label)

        self.empty_subtitle_label = QLabel(subtitle)
        self.empty_subtitle_label.setObjectName("libraryEmptySubtitle")
        self.empty_subtitle_label.setAlignment(Qt.AlignCenter)
        self.empty_subtitle_label.setWordWrap(True)
        empty_layout.addWidget(self.empty_subtitle_label)

        action_row = QHBoxLayout()
        action_row.addStretch()
        import_button = QPushButton("导入素材")
        import_button.setProperty("variant", "primary")
        import_button.setEnabled(self.image_db is not None)
        import_button.clicked.connect(self._on_import_btn_clicked)
        action_row.addWidget(import_button)
        refresh_button = QPushButton("刷新")
        refresh_button.setProperty("variant", "secondary")
        refresh_button.setEnabled(self.image_db is not None)
        refresh_button.clicked.connect(self.refresh)
        action_row.addWidget(refresh_button)
        action_row.addStretch()
        empty_layout.addLayout(action_row)

        self.thumbnail_layout.addWidget(self.empty_state, 0, 0, 1, 4)

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
        normalized = str(new_name).strip()
        if not normalized:
            QMessageBox.warning(self, "重命名失败", "名称不能为空")
            self.refresh()
            self.status_label.setText("名称不能为空")
            return

        if not self.image_db:
            self.status_label.setText("图像库未初始化")
            return

        try:
            self.image_db.update_image_metadata(img_id, {"name": normalized})
        except Exception as e:
            QMessageBox.warning(self, "重命名失败", f"无法重命名图片:\n{e}")
            self.refresh()
            self.status_label.setText("重命名失败")
            return

        # 这里不需要刷新整个列表，因为widget自己已经更新了文字
        self.status_label.setText(f"已重命名: {old_name} -> {normalized}")

    def _on_delete_requested_from_thumb(self, img_id):
        """缩略图请求删除"""
        self._delete_image(img_id)
        
    def _delete_image(self, img_id):
        """删除图像逻辑"""
        if QMessageBox.question(self, "确认删除", "确定要删除这张图片吗？") != QMessageBox.Yes:
            return
            
        if self.image_db:
            try:
                self.image_db.remove_image(img_id)
            except Exception as e:
                QMessageBox.warning(self, "删除失败", f"无法从库中删除图片:\n{e}")
                self.status_label.setText("删除失败")
                return
            self.refresh()
            self.status_label.setText("已删除图像")
        else:
            self.status_label.setText("图像库未初始化")
            
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
        if not self.image_db:
            QMessageBox.warning(self, "新建分组失败", "图像库未初始化")
            self.status_label.setText("图像库未初始化")
            return

        name, ok = QInputDialog.getText(self, "新建分组", "请输入分组名称:")
        if not ok:
            return

        normalized, error = self._validate_group_name(name)
        if error:
            QMessageBox.warning(self, "新建分组失败", error)
            self.status_label.setText(error)
            return

        try:
            self.image_db.add_group(normalized)
        except Exception as e:
            QMessageBox.warning(self, "新建分组失败", f"无法创建分组:\n{e}")
            self.status_label.setText("新建分组失败")
            return

        # 添加到下拉框
        self.group_combo.blockSignals(True)
        if self.group_combo.findText(normalized) == -1:
            self.group_combo.addItem(normalized)
        self.group_combo.setCurrentText(normalized)
        self.group_combo.blockSignals(False)

        # 刷新显示
        self._load_images(normalized)
        self.status_label.setText(f"已创建分组: {normalized}")

    def _on_group_context_menu(self, pos):
        """分组下拉框右键菜单"""
        current_group = self.group_combo.currentText()

        # 不允许删除 "全部" 和 "默认" 分组
        if current_group in RESERVED_GROUP_NAMES:
            return

        menu = QMenu(self)

        delete_action = menu.addAction(f"删除分组 '{current_group}'")
        delete_action.triggered.connect(lambda: self._delete_group(current_group))

        rename_action = menu.addAction(f"重命名分组 '{current_group}'")
        rename_action.triggered.connect(lambda: self._rename_group(current_group))

        menu.exec_(self.group_combo.mapToGlobal(pos))

    def _delete_group(self, group_name: str):
        """删除分组"""
        if group_name in RESERVED_GROUP_NAMES:
            return

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
            try:
                self.image_db.delete_group(group_name)
            except Exception as e:
                QMessageBox.warning(self, "删除分组失败", f"无法删除分组:\n{e}")
                self.status_label.setText("删除分组失败")
                return

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
        if old_name in RESERVED_GROUP_NAMES:
            return

        new_name, ok = QInputDialog.getText(self, "重命名分组", "请输入新名称:", text=old_name)
        if not ok:
            return

        normalized, error = self._validate_group_name(new_name, current_name=old_name)
        if error:
            QMessageBox.warning(self, "重命名分组失败", error)
            self.status_label.setText(error)
            return
        if normalized == old_name:
            return

        # 更新数据库
        if self.image_db:
            try:
                self.image_db.rename_group(old_name, normalized)
            except Exception as e:
                QMessageBox.warning(self, "重命名分组失败", f"无法重命名分组:\n{e}")
                self.status_label.setText("重命名分组失败")
                return

        # 更新下拉框
        index = self.group_combo.findText(old_name)
        if index >= 0:
            self.group_combo.setItemText(index, normalized)

        self.refresh()
        self.status_label.setText(f"已重命名分组: {old_name} -> {normalized}")

    def _normalize_group_name(self, name: str) -> str:
        """清理分组名称首尾空白。"""
        return str(name).strip()

    def _existing_group_names(self) -> List[str]:
        """返回当前下拉框中的分组名称。"""
        names = [
            self.group_combo.itemText(i)
            for i in range(self.group_combo.count())
        ]
        if self.image_db:
            try:
                names.extend(self.image_db.get_groups())
            except Exception:
                pass
        return names

    def _validate_group_name(self, name: str, current_name: Optional[str] = None):
        """校验分组名称，返回 (标准化名称, 错误信息)。"""
        normalized = self._normalize_group_name(name)
        if not normalized:
            return normalized, "分组名称不能为空"

        if normalized in RESERVED_GROUP_NAMES:
            return normalized, f"不能使用保留分组名: {normalized}"

        current_key = self._normalize_group_name(current_name).casefold() if current_name else None
        existing_keys = {
            self._normalize_group_name(group).casefold()
            for group in self._existing_group_names()
        }
        if current_key:
            existing_keys.discard(current_key)

        if normalized.casefold() in existing_keys:
            return normalized, f"分组已存在: {normalized}"

        return normalized, None

    def _on_rebuild_index(self):
        """重建所有图片的语义索引"""
        from PySide6.QtWidgets import QMessageBox, QProgressDialog

        if not self.image_db:
            QMessageBox.warning(self, "错误", "图像库未初始化")
            return
        if self._rebuild_thread is not None and self._rebuild_thread.isRunning():
            self.status_label.setText("索引正在重建中")
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
            progress.setAutoClose(False)
            progress.setAutoReset(False)

            worker = RebuildIndexThread(self.image_db)
            self._rebuild_thread = worker
            self._rebuild_progress = progress

            def update_progress(current: int, total: int):
                if progress.maximum() != total:
                    progress.setMaximum(total)
                progress.setValue(current)
                progress.setLabelText(f"正在处理: {current}/{total}")

            def handle_finished(rebuilt: int):
                QMessageBox.information(
                    self, "完成",
                    f"索引重建完成！\n成功处理 {rebuilt} 张图片。\n\n"
                    "现在可以使用中文语义搜索了，例如搜索\"杯子\"、\"风景\"等。"
                )
                self.refresh()

            def handle_canceled():
                self.status_label.setText("索引重建已取消")

            def handle_failed(message: str):
                QMessageBox.critical(self, "错误", f"重建索引失败: {message}")

            def cleanup():
                self.rebuild_btn.setEnabled(True)
                self.rebuild_btn.setText("重建索引")
                if self._rebuild_progress is progress:
                    progress.close()
                    progress.deleteLater()
                    self._rebuild_progress = None
                if self._rebuild_thread is worker:
                    self._rebuild_thread = None
                worker.deleteLater()

            progress.canceled.connect(worker.request_stop)
            worker.progress.connect(update_progress)
            worker.rebuild_finished.connect(handle_finished)
            worker.rebuild_canceled.connect(handle_canceled)
            worker.rebuild_failed.connect(handle_failed)
            worker.finished.connect(cleanup)
            worker.start()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"重建索引失败: {e}")
            self.rebuild_btn.setEnabled(True)
            self.rebuild_btn.setText("重建索引")

    def closeEvent(self, event):
        """关闭面板时等待索引重建线程自然退出。"""
        if self._rebuild_thread is not None and self._rebuild_thread.isRunning():
            self._rebuild_thread.request_stop()
            if not self._rebuild_thread.wait(5000):
                self.status_label.setText("索引仍在重建中，暂不能关闭")
                event.ignore()
                return
        super().closeEvent(event)

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
                normalized = self._normalize_group_name(group)
                if not normalized:
                    QMessageBox.warning(self, "导入失败", "分组名称不能为空")
                    self.status_label.setText("分组名称不能为空")
                    return
                if normalized not in groups:
                    normalized, error = self._validate_group_name(normalized)
                    if error:
                        QMessageBox.warning(self, "导入失败", error)
                        self.status_label.setText(error)
                        return
                self.import_requested.emit(files, normalized)

