"""
风格配方仓库
将调色参数沉淀为可复用的本地创作资产。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from .config import COLOR_PRESETS, DATA_DIR


@dataclass
class LookPreset:
    """可保存、可复用的调色风格配方。"""

    id: str
    name: str
    params: Dict[str, Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = "custom"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "source": self.source,
            "params": dict(self.params),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LookPreset":
        return cls(
            id=str(data.get("id") or f"look-{uuid4().hex[:10]}"),
            name=str(data.get("name") or "未命名风格"),
            description=str(data.get("description") or ""),
            tags=[str(tag) for tag in data.get("tags", [])],
            source=str(data.get("source") or "custom"),
            params=dict(data.get("params") or {}),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return slug or uuid4().hex[:8]


class LookPresetStore:
    """本地风格配方持久化仓库。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DATA_DIR / "look_presets.json"
        self._custom_presets: Dict[str, LookPreset] = {}
        self._load()

    def list_presets(self, include_builtin: bool = True) -> List[LookPreset]:
        """列出所有配方，内置配方排在自定义配方之前。"""
        presets: List[LookPreset] = []
        if include_builtin:
            presets.extend(self._builtin_presets())
        presets.extend(sorted(
            self._custom_presets.values(),
            key=lambda preset: preset.updated_at,
            reverse=True,
        ))
        return presets

    def get_preset(self, preset_id: str) -> Optional[LookPreset]:
        """按 id 查找配方。"""
        if preset_id in self._custom_presets:
            return self._custom_presets[preset_id]
        return next(
            (preset for preset in self._builtin_presets() if preset.id == preset_id),
            None,
        )

    def save_preset(
            self,
            name: str,
            params: Any,
            description: str = "",
            tags: Optional[Iterable[str]] = None) -> LookPreset:
        """新增或更新一个自定义风格配方。"""
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("风格配方名称不能为空")

        params_dict = params.to_dict() if hasattr(params, "to_dict") else dict(params)
        now = _now_iso()
        existing = self._find_custom_by_name(normalized)
        preset_id = existing.id if existing else f"custom-{_slugify(normalized)}-{uuid4().hex[:6]}"
        created_at = existing.created_at if existing else now

        preset = LookPreset(
            id=preset_id,
            name=normalized,
            description=description.strip(),
            tags=[str(tag).strip() for tag in tags or [] if str(tag).strip()],
            source="custom",
            params=params_dict,
            created_at=created_at,
            updated_at=now,
        )
        self._custom_presets[preset.id] = preset
        self._save()
        return preset

    def delete_preset(self, preset_id: str) -> bool:
        """删除自定义配方。内置配方不可删除。"""
        if preset_id not in self._custom_presets:
            return False
        del self._custom_presets[preset_id]
        self._save()
        return True

    def _find_custom_by_name(self, name: str) -> Optional[LookPreset]:
        key = name.casefold()
        return next(
            (preset for preset in self._custom_presets.values() if preset.name.casefold() == key),
            None,
        )

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._custom_presets = {}
            return

        presets = data.get("presets", []) if isinstance(data, dict) else []
        self._custom_presets = {}
        for item in presets:
            preset = LookPreset.from_dict(item)
            if preset.source == "custom":
                self._custom_presets[preset.id] = preset

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "presets": [preset.to_dict() for preset in self._custom_presets.values()],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _builtin_presets(self) -> List[LookPreset]:
        created_at = "builtin"
        presets = []
        for name, params in COLOR_PRESETS.items():
            presets.append(LookPreset(
                id=f"builtin-{_slugify(name)}",
                name=name,
                description="系统内置风格配方",
                tags=["内置"],
                source="builtin",
                params=dict(params),
                created_at=created_at,
                updated_at=created_at,
            ))
        return presets
