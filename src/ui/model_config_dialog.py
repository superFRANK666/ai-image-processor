"""
模型配置对话框。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.model_downloads import (
    DownloadedModel,
    clear_model_paths,
    clear_selected_models,
    download_selected_models,
    get_model_specs,
    get_target_path,
    config_model_path,
    is_model_downloaded,
    list_local_models,
    LLM_CONFIG_PATH,
)


CONFIG_GUIDE = """自选模型配置指南
1. 顶部“一句话调色后端”可直接配置禁用、本地模型、OpenAI、Anthropic 或 OpenAI 兼容 API，并写入 llm_config.json。
2. API 后端不需要下载本地 LLM；推荐在启动前设置服务商默认环境变量，避免把密钥写入配置文件。
3. 下面的模型列表负责下载和清理本地模型，路径与 scripts/download_all_models.py 保持一致。
4. 如果某一分类的目标路径已经有模型，确认下载会认为它已下载并跳过；要替换为自选模型，请先进入“清理模型”窗口勾选并删除对应本地模型。
5. 一句话调色本地模型建议选择 Qwen、Llama、Yi 等 Instruct/CausalLM 模型；下载后会自动写入 provider=local 的 llm_config.json。
6. 基础语义、CLIP 文本和 CLIP 图片编码模型需要兼容 SentenceTransformer；CLIP 文本与图片编码的向量维度最好一致。
7. 深度估计模型需要兼容 transformers 的 AutoModelForDepthEstimation；分割模型需要兼容 transformers 的 Sam2Model/Sam2Processor。
8. 自选模型体积可能很大，请确认磁盘空间和网络环境。下载失败时可修正模型 ID 后再次确认，已完成的目录会保留用于断点续传。"""


API_PROVIDER_DEFAULTS = {
    "openai": {
        "model": "",
        "base_url": "https://api.openai.com/v1",
        "api_key_required": True,
    },
    "anthropic": {
        "model": "",
        "base_url": "https://api.anthropic.com",
        "api_key_required": True,
    },
    "openai-compatible": {
        "model": "",
        "base_url": "",
        "api_key_required": False,
    },
}


MODEL_CONFIG_STYLE = """
QDialog#modelConfigDialog QRadioButton#modelOptionRadio {
    background-color: #121719;
    border: 1px solid #2d373c;
    border-radius: 6px;
    color: #dce8eb;
    font-weight: 700;
    min-height: 22px;
    padding: 8px 10px;
    spacing: 10px;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio:hover {
    background-color: #182124;
    border-color: #496067;
    color: #ffffff;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio:checked {
    background-color: #12342f;
    border-color: #18c7a7;
    color: #9ff8e5;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio:disabled {
    background-color: #15191b;
    border-color: #252c30;
    color: #657278;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #607078;
    border-radius: 3px;
    background-color: #090d0f;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio::indicator:hover {
    border-color: #8ba2aa;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio::indicator:checked {
    background-color: #18c7a7;
    border-color: #b8fff1;
}

QDialog#modelConfigDialog QRadioButton#modelOptionRadio::indicator:checked:disabled {
    background-color: #42635f;
    border-color: #607078;
}

QDialog#modelCleanupDialog {
    background-color: #0f1113;
}

QDialog#modelCleanupDialog QFrame#cleanupModelRow {
    background-color: #121719;
    border: 1px solid #2d373c;
    border-radius: 7px;
}

QDialog#modelCleanupDialog QFrame#cleanupModelRow[checked="true"] {
    background-color: #12342f;
    border-color: #18c7a7;
}

QDialog#modelCleanupDialog QCheckBox#cleanupModelCheck {
    color: #dce8eb;
    font-size: 13px;
    font-weight: 800;
    min-height: 22px;
    spacing: 10px;
}

QDialog#modelCleanupDialog QCheckBox#cleanupModelCheck:checked {
    color: #9ff8e5;
}

QDialog#modelCleanupDialog QCheckBox#cleanupModelCheck::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #607078;
    border-radius: 3px;
    background-color: #090d0f;
}

QDialog#modelCleanupDialog QCheckBox#cleanupModelCheck::indicator:hover {
    border-color: #8ba2aa;
}

QDialog#modelCleanupDialog QCheckBox#cleanupModelCheck::indicator:checked {
    background-color: #18c7a7;
    border-color: #b8fff1;
}

QDialog#modelCleanupDialog QLabel#cleanupModelMeta {
    color: #aab8bd;
    font-size: 12px;
}

QDialog#modelCleanupDialog QLabel#cleanupEmptyText {
    color: #9eb0b7;
    font-size: 13px;
    padding: 24px;
}
"""


class ModelConfigWorker(QThread):
    """后台执行模型下载或清理。"""

    progress = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, mode: str, selections: list[tuple[str, bool, str]] | list[Path], parent=None):
        super().__init__(parent)
        self.mode = mode
        self.selections = selections

    def run(self):
        try:
            if self.mode == "clear":
                results = clear_selected_models(self.selections, self.progress.emit)
                message = "\n".join(results) or "没有需要清理的模型。"
            elif self.mode == "clear_paths":
                results = clear_model_paths((Path(path) for path in self.selections), self.progress.emit)
                message = "\n".join(results) or "没有需要清理的模型。"
            else:
                results = download_selected_models(self.selections, self.progress.emit)
                message = "\n".join(results) or "没有需要下载的模型。"
            self.completed.emit(message)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelCleanupDialog(QDialog):
    """展示本地模型并让用户勾选需要删除的模型。"""

    def __init__(self, models: list[DownloadedModel], parent=None):
        super().__init__(parent)
        self.setObjectName("modelCleanupDialog")
        self.setWindowTitle("清理模型")
        self.setModal(True)
        self.resize(640, 560)
        self._models = list(models)
        self._rows = []
        self._setup_ui()
        self.setStyleSheet(MODEL_CONFIG_STYLE)

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("libraryManagerHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(3)

        title = QLabel("清理模型")
        title.setObjectName("libraryManagerTitle")
        subtitle = QLabel("选择要从本地 models 目录删除的模型。勾选后会高亮并显示对勾。")
        subtitle.setObjectName("libraryManagerSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        if self._models:
            for item in self._models:
                content_layout.addWidget(self._create_model_row(item))
            content_layout.addStretch()
        else:
            empty_label = QLabel("没有发现已下载到本地的模型。")
            empty_label.setObjectName("cleanupEmptyText")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setWordWrap(True)
            content_layout.addWidget(empty_label)
            content_layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)

        selection_layout = QHBoxLayout()
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(8)

        self.select_all_button = QPushButton("全选")
        self.select_all_button.setProperty("variant", "secondary")
        self.select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.select_all_button.setEnabled(bool(self._rows))
        selection_layout.addWidget(self.select_all_button)

        self.clear_selection_button = QPushButton("取消选择")
        self.clear_selection_button.setProperty("variant", "secondary")
        self.clear_selection_button.clicked.connect(lambda: self._set_all_checked(False))
        self.clear_selection_button.setEnabled(bool(self._rows))
        selection_layout.addWidget(self.clear_selection_button)

        selection_layout.addStretch()

        cancel_button = QPushButton("关闭")
        cancel_button.setProperty("variant", "secondary")
        cancel_button.clicked.connect(self.reject)
        selection_layout.addWidget(cancel_button)

        self.delete_button = QPushButton("删除选中")
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self.accept)
        self.delete_button.setEnabled(False)
        selection_layout.addWidget(self.delete_button)

        root_layout.addLayout(selection_layout)

    def _create_model_row(self, item: DownloadedModel):
        row = QFrame()
        row.setObjectName("cleanupModelRow")
        row.setProperty("checked", False)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        checkbox = QCheckBox(item.role_title)
        checkbox.setObjectName("cleanupModelCheck")
        checkbox.toggled.connect(
            lambda checked, box=checkbox, frame=row, model=item: self._set_row_checked(box, frame, model, checked)
        )

        model_label = QLabel(f"模型：{item.model_name} · 状态：{item.status}")
        model_label.setObjectName("cleanupModelMeta")
        model_label.setWordWrap(True)

        path_label = QLabel(f"路径：{item.path}")
        path_label.setObjectName("cleanupModelMeta")
        path_label.setWordWrap(True)

        layout.addWidget(checkbox)
        layout.addWidget(model_label)
        layout.addWidget(path_label)

        self._rows.append({"item": item, "checkbox": checkbox, "frame": row})
        return row

    def _set_row_checked(self, checkbox: QCheckBox, frame: QFrame, item: DownloadedModel, checked: bool):
        checkbox.setText(f"✓ {item.role_title}" if checked else item.role_title)
        frame.setProperty("checked", checked)
        frame.style().unpolish(frame)
        frame.style().polish(frame)
        frame.update()
        self._update_delete_button()

    def _set_all_checked(self, checked: bool):
        for row in self._rows:
            row["checkbox"].setChecked(checked)

    def _update_delete_button(self):
        self.delete_button.setEnabled(bool(self.selected_paths()))

    def selected_paths(self) -> list[Path]:
        return [
            row["item"].path
            for row in self._rows
            if row["checkbox"].isChecked()
        ]

    def selected_models(self) -> list[DownloadedModel]:
        return [
            row["item"]
            for row in self._rows
            if row["checkbox"].isChecked()
        ]


class ModelConfigDialog(QDialog):
    """设置菜单中的模型配置窗口。"""

    def __init__(self, parent=None, llm_config_path: Optional[Path] = None):
        super().__init__(parent)
        self.setObjectName("modelConfigDialog")
        self.setWindowTitle("模型配置")
        self.setModal(True)
        self.resize(720, 760)
        self._rows = []
        self._worker: Optional[ModelConfigWorker] = None
        self._llm_config_path = Path(llm_config_path) if llm_config_path else LLM_CONFIG_PATH
        self._loading_llm_config = False
        self._provider_radios = {}
        self._setup_ui()
        self.setStyleSheet(MODEL_CONFIG_STYLE)
        self._load_llm_runtime_config()
        self._refresh_status()

    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("libraryManagerHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(3)

        title = QLabel("模型配置")
        title.setObjectName("libraryManagerTitle")
        subtitle = QLabel("按项目功能选择默认模型或自定义模型，下载位置保持当前项目路径。")
        subtitle.setObjectName("libraryManagerSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root_layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(440)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        content_layout.addWidget(self._create_llm_backend_group())

        for spec in get_model_specs():
            content_layout.addWidget(self._create_model_group(spec))

        guide = QPlainTextEdit()
        guide.setReadOnly(True)
        guide.setPlainText(CONFIG_GUIDE)
        guide.setMinimumHeight(190)
        content_layout.addWidget(guide)
        content_layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        root_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("准备就绪")
        self.progress_label.setObjectName("mutedText")
        self.progress_label.setWordWrap(True)
        root_layout.addWidget(self.progress_label)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        self.clear_button = QPushButton("清理模型")
        self.clear_button.setProperty("variant", "danger")
        self.clear_button.clicked.connect(self._open_model_cleanup)
        button_layout.addWidget(self.clear_button)

        self.download_button = QPushButton("确认选择并下载缺失模型")
        self.download_button.setProperty("variant", "primary")
        self.download_button.clicked.connect(self._download_missing_models)
        button_layout.addWidget(self.download_button)
        root_layout.addLayout(button_layout)

    def _create_llm_backend_group(self):
        group = QGroupBox("一句话调色后端")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        hint = QLabel("选择自然语言调色使用的后端。API配置会直接保存到 llm_config.json；本地模型下载仍在下方模型列表完成。")
        hint.setObjectName("mutedText")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        provider_layout = QHBoxLayout()
        provider_layout.setContentsMargins(0, 0, 0, 0)
        provider_layout.setSpacing(8)
        provider_group = QButtonGroup(group)
        provider_options = [
            ("disabled", "禁用"),
            ("local", "本地模型"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("openai-compatible", "OpenAI兼容"),
        ]
        for provider, label in provider_options:
            radio = QRadioButton(label)
            radio.setObjectName("modelOptionRadio")
            radio.toggled.connect(lambda checked, name=provider: self._on_llm_provider_changed(name, checked))
            provider_group.addButton(radio)
            self._provider_radios[provider] = radio
            provider_layout.addWidget(radio)
        provider_layout.addStretch()
        layout.addLayout(provider_layout)

        self.api_model_input = self._create_labeled_input(layout, "模型名称", "")
        self.api_model_input.setObjectName("apiModelInput")
        self.api_base_url_input = self._create_labeled_input(layout, "Base URL", "")
        self.api_base_url_input.setObjectName("apiBaseUrlInput")
        self.api_key_input = self._create_labeled_input(layout, "API Key", "")
        self.api_key_input.setObjectName("apiKeyInput")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_timeout_input = self._create_labeled_input(layout, "超时秒数", "30")
        self.api_timeout_input.setObjectName("apiTimeoutInput")
        self.api_temperature_input = self._create_labeled_input(layout, "Temperature", "0.65，留空则不发送")
        self.api_temperature_input.setObjectName("apiTemperatureInput")
        self.api_max_tokens_input = self._create_labeled_input(layout, "Max tokens", "512，留空则使用后端默认")
        self.api_max_tokens_input.setObjectName("apiMaxTokensInput")

        self.llm_backend_status_label = QLabel()
        self.llm_backend_status_label.setObjectName("mutedText")
        self.llm_backend_status_label.setWordWrap(True)
        layout.addWidget(self.llm_backend_status_label)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addStretch()
        self.save_llm_config_button = QPushButton("保存一句话调色配置")
        self.save_llm_config_button.setProperty("variant", "primary")
        self.save_llm_config_button.clicked.connect(self._save_llm_runtime_config)
        action_layout.addWidget(self.save_llm_config_button)
        layout.addLayout(action_layout)

        return group

    def _create_labeled_input(self, parent_layout: QVBoxLayout, label_text: str, placeholder: str) -> QLineEdit:
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        label = QLabel(label_text)
        label.setObjectName("mutedText")
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        row.addWidget(label)
        row.addWidget(edit)
        parent_layout.addLayout(row)
        return edit

    def _read_raw_llm_config(self) -> dict:
        if not self._llm_config_path.exists():
            return {}
        try:
            data = json.loads(self._llm_config_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_llm_runtime_config(self):
        self._loading_llm_config = True
        config = self._read_raw_llm_config()
        enabled = bool(config.get("enabled", False))
        provider = str(config.get("provider", "local")).replace("_", "-")
        if not enabled:
            provider = "disabled"
        if provider not in self._provider_radios:
            provider = "local"

        self._provider_radios[provider].setChecked(True)

        if provider in API_PROVIDER_DEFAULTS:
            defaults = API_PROVIDER_DEFAULTS[provider]
            self.api_model_input.setText(str(config.get("model") or config.get("model_name") or defaults["model"]))
            self.api_base_url_input.setText(str(config.get("base_url") or defaults["base_url"]))
        else:
            self._apply_llm_provider_defaults("openai", replace=True)

        self.api_timeout_input.setText(str(config.get("timeout", 30)))
        temperature = config.get("temperature", 0.65)
        self.api_temperature_input.setText("" if temperature is None else str(temperature))
        max_tokens = config.get("max_tokens", 512)
        self.api_max_tokens_input.setText("" if max_tokens is None else str(max_tokens))
        self._loading_llm_config = False
        self._refresh_llm_backend_controls()

    def _selected_llm_provider(self) -> str:
        for provider, radio in self._provider_radios.items():
            if radio.isChecked():
                return provider
        return "local"

    def _on_llm_provider_changed(self, provider: str, checked: bool):
        if not checked:
            return
        if provider in API_PROVIDER_DEFAULTS and not self._loading_llm_config:
            self._apply_llm_provider_defaults(provider, replace=True)
        self._refresh_llm_backend_controls()

    def _apply_llm_provider_defaults(self, provider: str, replace: bool = False):
        defaults = API_PROVIDER_DEFAULTS.get(provider)
        if not defaults:
            return
        if replace or not self.api_model_input.text().strip():
            self.api_model_input.setText(defaults["model"])
        if replace or not self.api_base_url_input.text().strip():
            self.api_base_url_input.setText(defaults["base_url"])

    def _refresh_llm_backend_controls(self):
        provider = self._selected_llm_provider()
        api_enabled = provider in API_PROVIDER_DEFAULTS
        for widget in (
            self.api_model_input,
            self.api_base_url_input,
            self.api_key_input,
            self.api_timeout_input,
            self.api_temperature_input,
            self.api_max_tokens_input,
        ):
            widget.setEnabled(api_enabled)

        config_path = Path(self._llm_config_path)
        if provider == "disabled":
            text = f"当前将禁用大模型调色，配置保存到：{config_path}"
        elif provider == "local":
            text = f"当前将使用本地模型。请选择下方“一句话调色意图理解模型”并下载，配置保存到：{config_path}"
        else:
            text = f"当前将使用 {provider} API。保存后重启或重新加载语义模型生效，配置保存到：{config_path}"
        self.llm_backend_status_label.setText(text)

    def _api_float_value(self, edit: QLineEdit, default: Optional[float]) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{edit.placeholderText()} 不是有效数字") from exc

    def _api_int_value(self, edit: QLineEdit, default: Optional[int]) -> Optional[int]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{edit.placeholderText()} 不是有效整数") from exc

    def _local_llm_config_from_selection(self) -> dict:
        llm_row = next((row for row in self._rows if row["spec"].key == "llm_color"), None)
        if llm_row is None:
            raise RuntimeError("未找到一句话调色本地模型配置项")
        spec = llm_row["spec"]
        use_default = llm_row["default_radio"].isChecked()
        custom_value = llm_row["custom_input"].text().strip()
        if not use_default and not custom_value:
            raise ValueError("请填写一句话调色本地模型的自行配置内容，或改用默认模型。")
        target_path = get_target_path(spec, use_default, custom_value)
        return {
            "enabled": True,
            "provider": "local",
            "model_name": config_model_path(target_path),
            "device": "auto",
            "trust_remote_code": False,
        }

    def _save_llm_runtime_config(self):
        provider = self._selected_llm_provider()
        try:
            if provider == "disabled":
                config = {"enabled": False}
            elif provider == "local":
                config = self._local_llm_config_from_selection()
            else:
                model = self.api_model_input.text().strip()
                base_url = self.api_base_url_input.text().strip()
                if not model:
                    raise ValueError("请填写模型名称。")
                if not base_url:
                    raise ValueError("请填写 API Base URL。")

                existing = self._read_raw_llm_config()
                config = {
                    "enabled": True,
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "api_key_required": API_PROVIDER_DEFAULTS[provider]["api_key_required"],
                    "timeout": self._api_float_value(self.api_timeout_input, 30.0),
                    "temperature": self._api_float_value(self.api_temperature_input, 0.65),
                    "max_tokens": self._api_int_value(self.api_max_tokens_input, 512),
                }
                api_key = self.api_key_input.text().strip()
                if api_key:
                    config["api_key"] = api_key

                if existing.get("provider") == provider:
                    for key in ("endpoint", "headers", "api_version", "response_format", "extra_body"):
                        if key in existing:
                            config[key] = existing[key]

                config = {key: value for key, value in config.items() if value not in ("", None)}

            self._llm_config_path.parent.mkdir(parents=True, exist_ok=True)
            self._llm_config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.progress_label.setText(f"一句话调色配置已保存：{self._llm_config_path}")
            self._refresh_llm_backend_controls()
            QMessageBox.information(self, "配置已保存", "一句话调色配置已保存，重新加载语义模型或重启应用后生效。")
        except Exception as exc:
            QMessageBox.warning(self, "配置保存失败", str(exc))

    def _create_model_group(self, spec):
        group = QGroupBox(spec.role_title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(8)

        button_group = QButtonGroup(group)
        default_radio = QRadioButton(f"默认：{spec.default_model}")
        default_radio.setObjectName("modelOptionRadio")
        default_radio.setChecked(True)
        custom_radio = QRadioButton("自行配置")
        custom_radio.setObjectName("modelOptionRadio")
        button_group.addButton(default_radio)
        button_group.addButton(custom_radio)

        custom_input = QLineEdit()
        custom_input.setPlaceholderText(spec.custom_hint)
        custom_input.setEnabled(False)

        status_label = QLabel()
        status_label.setWordWrap(True)
        target_label = QLabel()
        target_label.setObjectName("mutedText")
        target_label.setWordWrap(True)

        layout.addWidget(default_radio)
        layout.addWidget(custom_radio)
        layout.addWidget(custom_input)
        layout.addWidget(status_label)
        layout.addWidget(target_label)

        row = {
            "spec": spec,
            "default_radio": default_radio,
            "custom_radio": custom_radio,
            "custom_input": custom_input,
            "status_label": status_label,
            "target_label": target_label,
        }
        self._rows.append(row)

        custom_radio.toggled.connect(lambda checked, edit=custom_input: edit.setEnabled(checked))
        custom_radio.toggled.connect(lambda _checked, current=row: self._refresh_row(current))
        default_radio.toggled.connect(lambda _checked, current=row: self._refresh_row(current))
        custom_input.textChanged.connect(lambda _text, current=row: self._refresh_row(current))

        return group

    def _row_selection(self, row) -> tuple[str, bool, str]:
        spec = row["spec"]
        use_default = row["default_radio"].isChecked()
        custom_value = row["custom_input"].text().strip()
        return (spec.key, use_default, custom_value)

    def _collect_selections(self, skip_llm_color: bool = False) -> Optional[list[tuple[str, bool, str]]]:
        selections = []
        for row in self._rows:
            spec = row["spec"]
            if skip_llm_color and spec.key == "llm_color":
                continue
            key, use_default, custom_value = self._row_selection(row)
            if not use_default and not custom_value:
                QMessageBox.warning(self, "缺少模型配置", f"请填写“{spec.role_title}”的自行配置内容。")
                row["custom_input"].setFocus()
                return None
            selections.append((key, use_default, custom_value))
        return selections

    def _refresh_status(self):
        for row in self._rows:
            self._refresh_row(row)

    def _refresh_row(self, row):
        spec = row["spec"]
        use_default = row["default_radio"].isChecked()
        custom_value = row["custom_input"].text().strip()
        target_path = get_target_path(spec, use_default, custom_value)

        downloaded = False
        if use_default or custom_value:
            downloaded = is_model_downloaded(spec, use_default, custom_value)

        status = "已下载" if downloaded else "未下载"
        if not spec.required:
            status += "（可选）"
        row["status_label"].setText(f"状态：{status}")
        row["target_label"].setText(f"保存位置：{Path(target_path)}")

    def _open_model_cleanup(self):
        dialog = ModelCleanupDialog(list_local_models(), self)
        if dialog.exec() != QDialog.Accepted:
            self._refresh_status()
            return

        selected_models = dialog.selected_models()
        selected_paths = dialog.selected_paths()
        if not selected_paths:
            return

        detail = "\n".join(f"- {item.role_title}：{item.path}" for item in selected_models)
        answer = QMessageBox.question(
            self,
            "确认清理模型",
            f"将删除以下本地模型文件：\n{detail}\n\n该操作不会删除源码，也不会移动模型目录以外的文件。是否继续？",
        )
        if answer != QMessageBox.Yes:
            return

        self._start_worker("clear_paths", selected_paths, "正在清理选中的模型...")

    def _download_missing_models(self):
        skip_llm_color = self._selected_llm_provider() != "local"
        selections = self._collect_selections(skip_llm_color=skip_llm_color)
        if selections is None:
            return
        if not selections:
            QMessageBox.information(self, "无需下载", "当前后端不需要下载本地模型。")
            return
        self._start_worker("download", selections, "正在下载缺失模型...")

    def _start_worker(self, mode: str, selections: list[tuple[str, bool, str]] | list[Path], initial_message: str):
        if self._worker and self._worker.isRunning():
            return

        self._set_busy(True)
        self.progress_label.setText(initial_message)
        self.progress_bar.setRange(0, 0)

        self._worker = ModelConfigWorker(mode, selections, self)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _set_busy(self, busy: bool):
        self.clear_button.setEnabled(not busy)
        self.download_button.setEnabled(not busy)
        self.save_llm_config_button.setEnabled(not busy)
        for radio in self._provider_radios.values():
            radio.setEnabled(not busy)
        for row in self._rows:
            row["default_radio"].setEnabled(not busy)
            row["custom_radio"].setEnabled(not busy)
            row["custom_input"].setEnabled(not busy and row["custom_radio"].isChecked())

        if not busy:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self._refresh_llm_backend_controls()
        else:
            for widget in (
                self.api_model_input,
                self.api_base_url_input,
                self.api_key_input,
                self.api_timeout_input,
                self.api_temperature_input,
                self.api_max_tokens_input,
            ):
                widget.setEnabled(False)

    def _on_worker_progress(self, message: str):
        self.progress_label.setText(message)

    def _on_worker_completed(self, message: str):
        self.progress_label.setText("操作完成")
        self._refresh_status()
        QMessageBox.information(self, "模型配置完成", message)

    def _on_worker_failed(self, message: str):
        self.progress_label.setText("操作失败")
        self._refresh_status()
        QMessageBox.warning(self, "模型配置失败", message)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "模型任务进行中", "请等待当前模型任务完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)
