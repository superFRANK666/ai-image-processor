"""
命令面板
提供现代桌面应用常见的 Ctrl+K 快速操作入口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


@dataclass(frozen=True)
class CommandDefinition:
    """一个可搜索、可触发的命令。"""

    id: str
    title: str
    subtitle: str = ""
    keywords: Iterable[str] = field(default_factory=tuple)
    callback: Optional[Callable[[], None]] = None
    enabled: bool = True


class CommandPalette(QDialog):
    """轻量命令面板。"""

    command_triggered = Signal(str)

    def __init__(self, commands: Iterable[CommandDefinition], parent=None):
        super().__init__(parent)
        self.setObjectName("commandPalette")
        self.setWindowTitle("命令面板")
        self.setModal(True)
        self.resize(560, 460)
        self._commands: List[CommandDefinition] = list(commands)
        self._setup_ui()
        self.set_commands(self._commands)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_group = QVBoxLayout()
        title_group.setContentsMargins(0, 0, 0, 0)
        title_group.setSpacing(2)
        title = QLabel("快速命令")
        title.setObjectName("paletteTitle")
        subtitle = QLabel("搜索操作、切换工作流或执行当前素材命令")
        subtitle.setObjectName("paletteSubtitle")
        title_group.addWidget(title)
        title_group.addWidget(subtitle)
        header.addLayout(title_group, 1)

        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "secondary")
        close_btn.clicked.connect(self.reject)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("paletteSearch")
        self.search_input.setPlaceholderText("输入命令、工作流或关键词")
        self.search_input.textChanged.connect(self._filter_commands)
        self.search_input.returnPressed.connect(self.trigger_current)
        layout.addWidget(self.search_input)

        self.command_list = QListWidget()
        self.command_list.setObjectName("paletteList")
        self.command_list.itemActivated.connect(self._trigger_item)
        layout.addWidget(self.command_list, 1)

        hint = QLabel("Enter 执行，Esc 关闭")
        hint.setObjectName("paletteHint")
        hint.setAlignment(Qt.AlignRight)
        layout.addWidget(hint)

    def set_commands(self, commands: Iterable[CommandDefinition]):
        """替换命令列表并刷新当前筛选结果。"""
        self._commands = list(commands)
        self._filter_commands(self.search_input.text() if hasattr(self, "search_input") else "")

    def focus_search(self):
        """聚焦搜索框，便于弹出后直接输入。"""
        self.search_input.setFocus()
        self.search_input.selectAll()

    def trigger_current(self):
        """执行当前选中的命令。"""
        self._trigger_item(self.command_list.currentItem())

    def _filter_commands(self, query: str):
        normalized = query.strip().casefold()
        self.command_list.clear()

        for command in self._commands:
            haystack = " ".join((
                command.title,
                command.subtitle,
                " ".join(command.keywords),
            )).casefold()
            if normalized and normalized not in haystack:
                continue
            item = QListWidgetItem(self._format_command(command))
            item.setData(Qt.UserRole, command.id)
            item.setToolTip(command.subtitle)
            if not command.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.command_list.addItem(item)

        if self.command_list.count():
            self.command_list.setCurrentRow(0)

    def _trigger_item(self, item: Optional[QListWidgetItem]):
        if item is None or not (item.flags() & Qt.ItemIsEnabled):
            return

        command_id = item.data(Qt.UserRole)
        command = next((entry for entry in self._commands if entry.id == command_id), None)
        if command is None or not command.enabled:
            return

        self.command_triggered.emit(command.id)
        self.accept()
        if command.callback is not None:
            command.callback()

    def _format_command(self, command: CommandDefinition) -> str:
        if not command.subtitle:
            return command.title
        return f"{command.title}\n{command.subtitle}"
