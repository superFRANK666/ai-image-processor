#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一模型下载脚本 - 完整版（包含所有必需模型）
支持的模型：
1. 一句话调色大语言模型 (默认 Qwen2.5-1.5B-Instruct，可自定义)
2. NLP理解模型 (paraphrase-multilingual-MiniLM-L12-v2)
3. 深度估计模型 (depth-anything-small)
4. SAM2 物体分割模型
5. CLIP 多语言文本检索模型
6. CLIP 图像编码器
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

# 修复Windows控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 设置环境变量,使用HuggingFace镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

PROJECT_ROOT = Path(__file__).parent.parent  # 项目根目录
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
LLM_CONFIG_PATH = PROJECT_ROOT / "llm_config.json"

DEFAULT_LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LLM_DIR_NAME = "Qwen2.5-1.5B-Instruct"
KNOWN_LLM_MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
    "qwen/qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
}


def prompt_input(prompt: str, default: str = "") -> str:
    """读取交互输入；在非交互环境中返回默认值。"""
    try:
        return input(prompt)
    except EOFError:
        print()
        return default


def model_id_from_hf_item(item) -> Optional[str]:
    """兼容 huggingface_hub 不同版本的模型条目字段。"""
    return getattr(item, "modelId", None) or getattr(item, "id", None)


def sanitize_model_dir_name(model_id: str) -> str:
    """将模型 ID 转换为安全的本地目录名。"""
    if model_id == DEFAULT_LLM_REPO_ID:
        return DEFAULT_LLM_DIR_NAME

    normalized = model_id.strip().replace("\\", "/")
    safe_name = normalized.replace("/", "__")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name).strip("._-")
    return safe_name or "custom-llm-model"


def config_model_path(local_dir: Path) -> str:
    """生成写入 llm_config.json 的模型路径。"""
    try:
        relative = local_dir.resolve().relative_to(PROJECT_ROOT.resolve())
        return f"./{relative.as_posix()}"
    except ValueError:
        return str(local_dir.resolve())


def write_llm_config(local_dir: Path):
    """写入一句话调色使用的 LLM 配置。"""
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
    print(f"  ✓ 已写入配置: {LLM_CONFIG_PATH}")
    print(f"  ✓ 一句话调色模型路径: {config['model_name']}")


def huggingface_endpoints():
    """返回用于查找和下载模型的 HuggingFace 端点列表。"""
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


def choose_best_model_id(model_ids, query: str) -> Optional[str]:
    """从搜索结果中选择最匹配的模型 ID。"""
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


def resolve_known_llm_alias(query: str) -> Optional[str]:
    """解析内置常见模型短名。"""
    return KNOWN_LLM_MODEL_ALIASES.get(query.casefold().strip())


def find_huggingface_model(query: str) -> Optional[str]:
    """按用户输入查找 HuggingFace 模型。"""
    from huggingface_hub import HfApi

    normalized_query = query.strip()
    if not normalized_query:
        return None

    known_model = resolve_known_llm_alias(normalized_query)
    if known_model:
        print(f"  ✓ 找到模型: {known_model}")
        return known_model

    last_error = None
    for endpoint in huggingface_endpoints():
        api = HfApi(endpoint=endpoint)

        exact_candidates = [normalized_query]
        if "/" not in normalized_query and normalized_query.casefold().startswith("qwen"):
            exact_candidates.insert(0, f"Qwen/{normalized_query}")

        for candidate in exact_candidates:
            if "/" not in candidate:
                continue
            try:
                info = api.model_info(candidate)
                found = model_id_from_hf_item(info) or candidate
                print(f"  ✓ 找到模型: {found}")
                return found
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
        model_ids = [model_id for model_id in model_ids if model_id]
        selected_model = choose_best_model_id(model_ids, normalized_query)
        if selected_model:
            print(f"  ✓ 找到模型: {selected_model}")
            return selected_model

    if last_error:
        print(f"  ⚠ 模型搜索失败或无结果: {last_error}")
    return None


def print_manual_llm_download_help(requested_model: str, target_dir: Path, not_found: bool = True):
    """提示用户手动下载未能自动找到的模型。"""
    if not_found:
        print(f"  ✗ 未找到模型: {requested_model}")
        print("  请确认模型名称是否为 HuggingFace 仓库 ID，例如: Qwen/Qwen2.5-1.5B-Instruct")
    else:
        print(f"  ✗ 无法自动下载模型: {requested_model}")
        print("  请检查网络、磁盘空间，或稍后重试。")
    print("  如需手动下载，请将完整模型仓库文件保存到:")
    print(f"    {target_dir}")
    print("  手动下载示例:")
    print(
        "    python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id='模型ID', local_dir=r'{target_dir}', "
        "local_dir_use_symlinks=False)\""
    )
    print("  下载完成后，可将 llm_config.json 设置为:")
    manual_config = {
        "enabled": True,
        "provider": "local",
        "model_name": config_model_path(target_dir),
        "device": "auto",
        "trust_remote_code": False,
    }
    print(json.dumps(manual_config, ensure_ascii=False, indent=4))


def llm_snapshot_ready(model_dir: Path) -> bool:
    """判断本地大语言模型目录是否已包含可加载的基础文件。"""
    if not model_dir.exists() or not model_dir.is_dir():
        return False

    has_config = (model_dir / "config.json").exists()
    has_tokenizer = any(
        (model_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    has_weights = any(model_dir.glob("*.safetensors")) or any(model_dir.glob("*.bin"))
    return has_config and has_tokenizer and has_weights


def download_llm_snapshot(model_id: str, save_path: Path):
    """下载大语言模型仓库到本地目录。"""
    from huggingface_hub import snapshot_download

    if llm_snapshot_ready(save_path):
        print(f"  ✓ 模型目录已存在,跳过下载: {save_path}")
        return
    if save_path.exists() and any(save_path.iterdir()):
        print("  ⚠ 检测到未完整的模型目录，将尝试断点续传")

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"  正在下载模型仓库: {model_id}")
    print(f"  保存位置: {save_path}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(save_path),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )
    print(f"  ✓ 下载完成: {save_path}")


def configure_llm_model() -> bool:
    """在其它模型下载前配置一句话调色大语言模型。"""
    print("\n" + "=" * 80)
    print("[1/6] 配置一句话调色大语言模型")
    print("=" * 80)
    print("一句话调色需要一个本地大语言模型来理解自然语言调色意图。")
    print("请选择要下载和配置的模型:")
    print(f"  1. {DEFAULT_LLM_REPO_ID} (默认推荐, 约3GB)")
    print("  2. Others (自行填写 HuggingFace 模型名称)")

    choice = prompt_input("请选择 (1/2, 默认1): ", default="1").strip()
    if choice in ("", "1"):
        model_id = DEFAULT_LLM_REPO_ID
        local_dir = MODELS_DIR / DEFAULT_LLM_DIR_NAME
    elif choice == "2":
        requested_model = prompt_input("请输入模型名称或 HuggingFace 仓库ID: ").strip()
        if not requested_model:
            print("  ✗ 未输入模型名称，跳过 LLM 配置")
            return False

        print(f"  正在查找模型: {requested_model}")
        model_id = find_huggingface_model(requested_model)
        local_dir = MODELS_DIR / sanitize_model_dir_name(model_id or requested_model)
        if not model_id:
            print_manual_llm_download_help(requested_model, local_dir)
            return False
    else:
        print("  输入无效，使用默认模型")
        model_id = DEFAULT_LLM_REPO_ID
        local_dir = MODELS_DIR / DEFAULT_LLM_DIR_NAME

    try:
        download_llm_snapshot(model_id, local_dir)
        write_llm_config(local_dir)
        return True
    except Exception as e:
        print(f"  ✗ 大语言模型下载或配置失败: {e}")
        print_manual_llm_download_help(model_id, local_dir, not_found=False)
        return False


print("=" * 80)
print("   AI影像处理软件 - 完整模型下载工具")
print("=" * 80)
print("\n📦 将下载以下模型:")
print("  [建议] 1. 一句话调色大语言模型 (默认Qwen2.5-1.5B, ~3GB)")
print("  [必需] 2. NLP理解模型 (~471MB)")
print("  [必需] 3. 深度估计模型 (~99MB)")
print("  [必需] 4. SAM2物体分割模型 (~155MB)")
print("  [必需] 5. CLIP多语言模型 (~540MB)")
print("  [必需] 6. CLIP图像编码器 (~600MB)")
print("\n⏱️  预计总下载时间: 10-30分钟 (取决于网络速度)")
print("=" * 80)

# 下载进度统计
total_models = 6
downloaded_models = 0
failed_models = []

if configure_llm_model():
    downloaded_models += 1
else:
    failed_models.append(("一句话调色大语言模型", "未完成自动下载或配置"))

# 2. 下载NLP理解模型
print("\n" + "=" * 80)
print("[2/6] 下载NLP理解模型 (约471MB)...")
print("=" * 80)
try:
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    save_path = MODELS_DIR / "paraphrase-multilingual-MiniLM-L12-v2"
    if save_path.exists():
        print("  ✓ 模型已存在,跳过")
        downloaded_models += 1
    else:
        model = SentenceTransformer(model_name)
        model.save(str(save_path))
        print(f"  ✓ 下载完成: {save_path}")
        downloaded_models += 1
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("NLP理解模型", str(e)))

# 3. 下载深度估计模型
print("\n" + "=" * 80)
print("[3/6] 下载深度估计模型 (约99MB)...")
print("=" * 80)
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_name = "LiheYoung/depth-anything-small-hf"
    save_path = MODELS_DIR / "depth-anything-small"
    if save_path.exists():
        print("  ✓ 模型已存在,跳过")
        downloaded_models += 1
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        processor.save_pretrained(str(save_path))
        model.save_pretrained(str(save_path))
        print(f"  ✓ 下载完成: {save_path}")
        downloaded_models += 1
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("深度估计模型", str(e)))

# 4. 下载SAM2物体分割模型
print("\n" + "=" * 80)
print("[4/6] 下载SAM2物体分割模型 (约155MB)...")
print("=" * 80)
try:
    from huggingface_hub import snapshot_download
    sam2_dir = MODELS_DIR / "sam2-hiera-tiny"

    has_config = (sam2_dir / "config.json").exists()
    has_processor = (sam2_dir / "preprocessor_config.json").exists() or (sam2_dir / "processor_config.json").exists()
    has_weights = any(sam2_dir.glob("*.safetensors")) or any(sam2_dir.glob("*.bin"))
    if has_config and has_processor and has_weights:
        print("  ✓ 模型已存在,跳过")
        downloaded_models += 1
    else:
        last_error = None
        for endpoint in huggingface_endpoints():
            try:
                print(f"  正在下载: {endpoint}")
                snapshot_download(
                    repo_id="facebook/sam2-hiera-tiny",
                    local_dir=str(sam2_dir),
                    max_workers=4,
                    endpoint=endpoint,
                )
                print(f"  ✓ 下载完成: {sam2_dir}")
                downloaded_models += 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(f"  下载源失败: {endpoint} ({exc})")
        if last_error:
            raise last_error
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("SAM2物体分割模型", str(e)))

# 5. 下载多语言CLIP模型
print("\n" + "=" * 80)
print("[5/6] 下载多语言CLIP模型 (约540MB)...")
print("=" * 80)
try:
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    save_path = MODELS_DIR / "clip-ViT-B-32-multilingual-v1"
    if save_path.exists():
        print("  ✓ 模型已存在,跳过")
        downloaded_models += 1
    else:
        model = SentenceTransformer(model_name)
        model.save(str(save_path))
        print(f"  ✓ 下载完成: {save_path}")
        downloaded_models += 1
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("CLIP多语言模型", str(e)))

# 6. 下载原始CLIP图像编码器
print("\n" + "=" * 80)
print("[6/6] 下载CLIP图像编码器 (约600MB)...")
print("=" * 80)
try:
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/clip-ViT-B-32"
    save_path = MODELS_DIR / "clip-ViT-B-32"
    if save_path.exists():
        print("  ✓ 模型已存在,跳过")
        downloaded_models += 1
    else:
        model = SentenceTransformer(model_name)
        model.save(str(save_path))
        print(f"  ✓ 下载完成: {save_path}")
        downloaded_models += 1
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("CLIP图像编码器", str(e)))

# 最终统计
print("\n" + "=" * 80)
print("模型下载流程完成!")
print("=" * 80)

print(f"\n✓ 成功下载: {downloaded_models}/{total_models} 个模型")

if failed_models:
    print(f"\n✗ 失败的模型 ({len(failed_models)}):")
    for model_name, error in failed_models:
        print(f"  - {model_name}: {error}")
    print("\n建议:")
    print("  1. 检查网络连接")
    print("  2. 重新运行此脚本（支持断点续传）")
    print("  3. 或使用各模型的独立下载脚本")
else:
    print("\n🎉 所有模型下载成功!")

print("\n📁 模型目录: ", MODELS_DIR)
print("\n接下来:")
print("  1. 确认 models 目录下的模型文件")
print("  2. 运行 python main.py 启动程序")
print("  3. 一句话调色大语言模型配置位于 llm_config.json")
print("  4. 使用 src/utils/model_checksum.py 验证模型完整性")
print("\n" + "=" * 80)
