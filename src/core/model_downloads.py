"""
模型下载清单与工具函数。

这里的默认模型与保存位置保持和 scripts/download_all_models.py 一致，供设置窗口复用。
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LLM_CONFIG_PATH = PROJECT_ROOT / "llm_config.json"

DEFAULT_LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LLM_DIR_NAME = "Qwen2.5-1.5B-Instruct"
KNOWN_LLM_MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
    "qwen/qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
}


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ModelSpec:
    """一个可配置模型项。"""

    key: str
    role_title: str
    default_model: str
    kind: str
    target_path: Path
    custom_hint: str
    guide: str
    required: bool = True


@dataclass(frozen=True)
class DownloadedModel:
    """本地已存在的模型目录或文件。"""

    key: str
    role_title: str
    model_name: str
    path: Path
    status: str
    known: bool = True


MODEL_SPECS = (
    ModelSpec(
        key="llm_color",
        role_title="一句话调色意图理解模型",
        default_model=DEFAULT_LLM_REPO_ID,
        kind="llm",
        target_path=MODELS_DIR / DEFAULT_LLM_DIR_NAME,
        custom_hint="HuggingFace 仓库 ID，例如 Qwen/Qwen2.5-3B-Instruct",
        guide="用于理解自然语言调色描述。建议选择 Instruct/CausalLM 类模型；下载后会写入 llm_config.json。",
    ),
    ModelSpec(
        key="nlp_parser",
        role_title="基础中文语义解析模型",
        default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        kind="sentence_transformer",
        target_path=MODELS_DIR / "paraphrase-multilingual-MiniLM-L12-v2",
        custom_hint="SentenceTransformer 兼容模型 ID",
        guide="用于关键词与句向量理解。自选模型需要兼容 SentenceTransformer，并能输出文本向量。",
    ),
    ModelSpec(
        key="depth_estimation",
        role_title="3D 生成深度估计模型",
        default_model="LiheYoung/depth-anything-small-hf",
        kind="depth",
        target_path=MODELS_DIR / "depth-anything-small",
        custom_hint="兼容 AutoModelForDepthEstimation 的 HuggingFace 模型 ID",
        guide="用于图片到 3D 的深度估计。自选模型需要支持 transformers 的深度估计接口。",
    ),
    ModelSpec(
        key="sam2_segmenter",
        role_title="SAM2 物体分割模型",
        default_model="facebook/sam2-hiera-tiny",
        kind="hf_snapshot",
        target_path=MODELS_DIR / "sam2-hiera-tiny",
        custom_hint="SAM2 兼容 HuggingFace 仓库 ID，例如 facebook/sam2-hiera-small",
        guide="用于点选、框选、路径选择、主体遮罩和物体 3D 工作流。自选模型需要兼容 Transformers Sam2Model/Sam2Processor。",
    ),
    ModelSpec(
        key="clip_text",
        role_title="图像库多语言文本检索模型",
        default_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        kind="sentence_transformer",
        target_path=MODELS_DIR / "clip-ViT-B-32-multilingual-v1",
        custom_hint="多语言 CLIP 文本模型 ID",
        guide="用于中文/英文文本检索图库。自选模型需要和图像编码器处于兼容向量空间。",
    ),
    ModelSpec(
        key="clip_image",
        role_title="图像库图片编码模型",
        default_model="sentence-transformers/clip-ViT-B-32",
        kind="sentence_transformer",
        target_path=MODELS_DIR / "clip-ViT-B-32",
        custom_hint="CLIP 图像编码 SentenceTransformer 模型 ID",
        guide="用于图库图片向量编码。自选模型应与多语言文本模型维度一致，否则检索会退回传统特征。",
    ),
)


def get_model_specs() -> list[ModelSpec]:
    """返回设置窗口展示的模型清单。"""
    return list(MODEL_SPECS)


def model_spec_by_key(key: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"未知模型配置项: {key}")


def ensure_hf_endpoint():
    """保持下载脚本既有默认镜像行为。"""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def sanitize_model_dir_name(model_id: str) -> str:
    """将模型 ID 转换为安全的本地目录名，与下载脚本规则一致。"""
    if model_id == DEFAULT_LLM_REPO_ID:
        return DEFAULT_LLM_DIR_NAME

    normalized = model_id.strip().replace("\\", "/")
    safe_name = normalized.replace("/", "__")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name).strip("._-")
    return safe_name or "custom-llm-model"


def config_model_path(local_dir: Path) -> str:
    try:
        relative = local_dir.resolve().relative_to(PROJECT_ROOT.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(local_dir.resolve())


def write_llm_config(local_dir: Path):
    config = {
        "enabled": True,
        "provider": "local",
        "model_name": config_model_path(local_dir),
        "device": "auto",
        "trust_remote_code": False,
    }
    LLM_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_known_llm_alias(query: str) -> Optional[str]:
    return KNOWN_LLM_MODEL_ALIASES.get(query.casefold().strip())


def huggingface_endpoints() -> list[str]:
    endpoints = []
    env_endpoint = os.environ.get("HF_ENDPOINT")
    if env_endpoint:
        endpoints.append(env_endpoint.rstrip("/"))
    endpoints.append("https://huggingface.co")

    unique = []
    for endpoint in endpoints:
        if endpoint not in unique:
            unique.append(endpoint)
    return unique


def model_id_from_hf_item(item) -> Optional[str]:
    return getattr(item, "modelId", None) or getattr(item, "id", None)


def choose_best_model_id(model_ids: Iterable[str], query: str) -> Optional[str]:
    model_ids = list(model_ids)
    if not model_ids:
        return None

    normalized_query = query.casefold().strip()
    for model_id in model_ids:
        if model_id.casefold() == normalized_query:
            return model_id

    for model_id in model_ids:
        short_name = model_id.rsplit("/", 1)[-1]
        if short_name.casefold() == normalized_query:
            return model_id

    return model_ids[0]


def find_huggingface_model(query: str, progress: Optional[ProgressCallback] = None) -> Optional[str]:
    """按用户输入查找 HuggingFace 模型，复用下载脚本的匹配策略。"""
    from huggingface_hub import HfApi

    normalized_query = query.strip()
    if not normalized_query:
        return None

    known_model = resolve_known_llm_alias(normalized_query)
    if known_model:
        return known_model

    last_error = None
    for endpoint in huggingface_endpoints():
        api = HfApi(endpoint=endpoint)
        if progress:
            progress(f"正在查找模型: {normalized_query} ({endpoint})")

        exact_candidates = [normalized_query]
        if "/" not in normalized_query and normalized_query.casefold().startswith("qwen"):
            exact_candidates.insert(0, f"Qwen/{normalized_query}")

        for candidate in exact_candidates:
            if "/" not in candidate:
                continue
            try:
                info = api.model_info(candidate)
                return model_id_from_hf_item(info) or candidate
            except Exception as exc:
                last_error = exc

        try:
            try:
                results = list(
                    api.list_models(
                        search=normalized_query,
                        sort="downloads",
                        direction=-1,
                        limit=10,
                    )
                )
            except TypeError:
                results = list(api.list_models(search=normalized_query, limit=10))
        except Exception as exc:
            last_error = exc
            continue

        model_ids = [model_id_from_hf_item(item) for item in results]
        selected_model = choose_best_model_id((model_id for model_id in model_ids if model_id), normalized_query)
        if selected_model:
            return selected_model

    if progress and last_error:
        progress(f"模型搜索失败: {last_error}")
    return None


def get_target_path(spec: ModelSpec, use_default: bool, custom_value: str = "") -> Path:
    """返回该配置项的实际保存位置。"""
    if spec.kind == "llm":
        if use_default or not custom_value.strip():
            return MODELS_DIR / DEFAULT_LLM_DIR_NAME
        model_id = resolve_known_llm_alias(custom_value.strip()) or custom_value.strip()
        return MODELS_DIR / sanitize_model_dir_name(model_id)
    return spec.target_path


def llm_snapshot_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    has_config = (model_dir / "config.json").exists()
    has_tokenizer = any(
        (model_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_config and has_tokenizer and has_weights


def sentence_transformer_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    return (model_dir / "modules.json").exists() or any(model_dir.rglob("*.safetensors"))


def depth_model_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_config and has_weights


def hf_snapshot_ready(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_processor = (
        (model_dir / "preprocessor_config.json").exists()
        or (model_dir / "processor_config.json").exists()
    )
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_config and has_processor and has_weights


def is_model_downloaded(spec: ModelSpec, use_default: bool = True, custom_value: str = "") -> bool:
    target = get_target_path(spec, use_default, custom_value)
    if spec.kind == "llm":
        return llm_snapshot_ready(target)
    if spec.kind == "sentence_transformer":
        return sentence_transformer_ready(target)
    if spec.kind == "depth":
        return depth_model_ready(target)
    if spec.kind == "hf_snapshot":
        return hf_snapshot_ready(target)
    return target.exists()


def local_model_status(spec: ModelSpec, path: Path) -> str:
    """返回本地模型路径状态，供清理窗口展示。"""
    if not path.exists():
        return "不存在"
    if is_model_downloaded(spec):
        return "完整"
    return "未完整"


def list_local_models() -> list[DownloadedModel]:
    """列出 models 目录下已经存在的模型，包含未完整下载的残留目录。"""
    items = []
    known_paths = set()

    for spec in MODEL_SPECS:
        target_path = get_target_path(spec, True, "")
        known_paths.add(target_path.resolve())
        if not target_path.exists():
            continue
        items.append(
            DownloadedModel(
                key=spec.key,
                role_title=spec.role_title,
                model_name=spec.default_model,
                path=target_path,
                status=local_model_status(spec, target_path),
                known=True,
            )
        )

    if MODELS_DIR.exists():
        for child in sorted(MODELS_DIR.iterdir(), key=lambda path: path.name.casefold()):
            resolved = child.resolve()
            if resolved in known_paths:
                continue
            items.append(
                DownloadedModel(
                    key=f"local::{child.name}",
                    role_title="未归类本地模型",
                    model_name=child.name,
                    path=child,
                    status="本地存在",
                    known=False,
                )
            )

    return items


def download_llm_snapshot(model_id: str, save_path: Path, progress: Optional[ProgressCallback] = None):
    from huggingface_hub import snapshot_download

    if llm_snapshot_ready(save_path):
        if progress:
            progress(f"已存在，跳过: {save_path}")
        write_llm_config(save_path)
        return

    save_path.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"正在下载大语言模型: {model_id}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )
    write_llm_config(save_path)


def download_sentence_transformer(model_id: str, save_path: Path, progress: Optional[ProgressCallback] = None):
    from sentence_transformers import SentenceTransformer

    if sentence_transformer_ready(save_path):
        if progress:
            progress(f"已存在，跳过: {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"正在下载 SentenceTransformer 模型: {model_id}")
    model = SentenceTransformer(model_id)
    model.save(str(save_path))


def download_depth_model(model_id: str, save_path: Path, progress: Optional[ProgressCallback] = None):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if depth_model_ready(save_path):
        if progress:
            progress(f"已存在，跳过: {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"正在下载深度估计模型: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    processor.save_pretrained(str(save_path))
    model.save_pretrained(str(save_path))


def download_hf_snapshot(model_id: str, save_path: Path, progress: Optional[ProgressCallback] = None):
    from huggingface_hub import snapshot_download

    if hf_snapshot_ready(save_path):
        if progress:
            progress(f"已存在，跳过: {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    last_error = None
    for endpoint in huggingface_endpoints():
        try:
            if progress:
                progress(f"正在下载模型仓库: {model_id} ({endpoint})")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(save_path),
                max_workers=4,
                endpoint=endpoint,
            )
            return
        except Exception as exc:
            last_error = exc
            if progress:
                progress(f"下载源失败: {endpoint} ({exc})")

    raise RuntimeError(f"无法下载模型仓库 {model_id}: {last_error}")


def download_model(spec: ModelSpec, use_default: bool, custom_value: str = "", progress: Optional[ProgressCallback] = None):
    ensure_hf_endpoint()
    MODELS_DIR.mkdir(exist_ok=True)
    target_path = get_target_path(spec, use_default, custom_value)
    model_id = spec.default_model if use_default else custom_value.strip()

    if is_model_downloaded(spec, use_default, custom_value):
        if progress:
            progress(f"{spec.role_title} 已下载，跳过")
        if spec.kind == "llm":
            write_llm_config(target_path)
        return

    if spec.kind == "llm":
        if use_default:
            model_id = DEFAULT_LLM_REPO_ID
        else:
            resolved = find_huggingface_model(model_id, progress)
            if not resolved:
                raise RuntimeError(f"未找到模型: {model_id}")
            model_id = resolved
            target_path = MODELS_DIR / sanitize_model_dir_name(model_id)
        download_llm_snapshot(model_id, target_path, progress)
        return

    if spec.kind == "sentence_transformer":
        download_sentence_transformer(model_id, target_path, progress)
        return

    if spec.kind == "depth":
        download_depth_model(model_id, target_path, progress)
        return

    if spec.kind == "hf_snapshot":
        download_hf_snapshot(model_id, target_path, progress)
        return

    raise RuntimeError(f"不支持的模型类型: {spec.kind}")


def _assert_safe_model_target(target_path: Path):
    models_root = MODELS_DIR.resolve()
    resolved = target_path.resolve()
    if models_root == resolved or models_root not in resolved.parents:
        raise ValueError(f"拒绝清理 models 目录以外的路径: {target_path}")


def clear_model(spec: ModelSpec, use_default: bool, custom_value: str = "", progress: Optional[ProgressCallback] = None) -> bool:
    target_path = get_target_path(spec, use_default, custom_value)
    return clear_model_path(target_path, progress)


def clear_model_path(target_path: Path, progress: Optional[ProgressCallback] = None) -> bool:
    """清理指定模型路径；目标必须位于 models 目录内。"""
    _assert_safe_model_target(target_path)

    if not target_path.exists():
        if progress:
            progress(f"未发现已下载文件: {target_path}")
        return False

    if target_path.is_file():
        target_path.unlink()
        if progress:
            progress(f"已清除: {target_path}")
        return True

    shutil.rmtree(target_path)
    if progress:
        progress(f"已清除: {target_path}")
    return True


def clear_model_paths(paths: Iterable[Path], progress: Optional[ProgressCallback] = None) -> list[str]:
    """批量清理用户在清理窗口中勾选的模型路径。"""
    results = []
    for path in paths:
        removed = clear_model_path(Path(path), progress)
        results.append(f"{Path(path).name}: {'已清除' if removed else '无需清除'}")
    return results


def download_selected_models(
    selections: Iterable[tuple[str, bool, str]],
    progress: Optional[ProgressCallback] = None,
) -> list[str]:
    results = []
    for key, use_default, custom_value in selections:
        spec = model_spec_by_key(key)
        if progress:
            progress(f"开始处理: {spec.role_title}")
        download_model(spec, use_default, custom_value, progress)
        results.append(f"{spec.role_title}: 完成")
    return results


def clear_selected_models(
    selections: Iterable[tuple[str, bool, str]],
    progress: Optional[ProgressCallback] = None,
) -> list[str]:
    results = []
    for key, use_default, custom_value in selections:
        spec = model_spec_by_key(key)
        removed = clear_model(spec, use_default, custom_value, progress)
        results.append(f"{spec.role_title}: {'已清除' if removed else '无需清除'}")
    return results
