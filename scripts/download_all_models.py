#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一模型下载脚本 - 完整版（包含所有必需模型）
支持的模型：
1. 一句话调色大语言模型 (默认 Qwen2.5-1.5B-Instruct，可自定义)
2. NLP理解模型 (paraphrase-multilingual-MiniLM-L12-v2)
3. 深度估计模型 (depth-anything-small)
4. MobileSAM 分割模型
5. CLIP 多语言图像检索模型
6. SAM2 高精度分割模型 (可选)
"""
import os
import sys
import json
import hashlib
import re
import shutil
import zipfile
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

MOBILE_SAM_SHA256 = "f3c0d8cda613564d499310dab6c812cd80d9de20dd0e7d7b3ea0cd86ff5c76d6"
DEFAULT_LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LLM_DIR_NAME = "Qwen2.5-1.5B-Instruct"
KNOWN_LLM_MODEL_ALIASES = {
    "qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
    "qwen/qwen2.5-1.5b-instruct": DEFAULT_LLM_REPO_ID,
}


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """校验文件 SHA256。"""
    actual_sha256 = calculate_sha256(file_path)
    if actual_sha256.lower() != expected_sha256.lower():
        print("  ✗ SHA256 校验失败")
        print(f"    预期: {expected_sha256}")
        print(f"    实际: {actual_sha256}")
        return False
    print(f"  ✓ SHA256 校验通过: {actual_sha256}")
    return True


def download_file_atomic(url: str, output_path: Path, expected_sha256: str = None, stream: bool = True):
    """下载到临时文件，校验通过后再替换目标文件。"""
    import requests

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    response = requests.get(url, timeout=60, stream=stream)
    response.raise_for_status()

    if stream:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  下载进度: {percent:.1f}%", end='')
        if total_size > 0:
            print()
    else:
        with open(tmp_path, 'wb') as f:
            f.write(response.content)

    if expected_sha256 and not verify_sha256(tmp_path, expected_sha256):
        tmp_path.unlink(missing_ok=True)
        raise ValueError("下载文件校验失败")

    tmp_path.replace(output_path)


def safe_extract_zip(zip_path: Path, target_dir: Path):
    """安全解压 ZIP，拒绝路径穿越条目。"""
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            destination = (target_root / member.filename).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise ValueError(f"ZIP 包含非法路径: {member.filename}")
        zip_ref.extractall(target_root)


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
print("  [必需] 4. MobileSAM分割模型 (~40MB)")
print("  [必需] 5. CLIP多语言模型 (~540MB)")
print("  [可选] 6. SAM2高精度分割 (~155MB)")
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

# 4. 下载MobileSAM权重
print("\n" + "=" * 80)
print("[4/6] 下载MobileSAM权重 (约40MB)...")
print("=" * 80)
try:
    import requests

    sam_dir = MODELS_DIR / "mobile-sam"
    sam_dir.mkdir(exist_ok=True)
    sam_file = sam_dir / "mobile_sam.pt"

    if sam_file.exists() and verify_sha256(sam_file, MOBILE_SAM_SHA256):
        print("  ✓ 模型已存在且校验通过,跳过")
        downloaded_models += 1
    else:
        if sam_file.exists():
            print("  ⚠ 已有模型校验失败，将重新下载")
            sam_file.unlink()

        url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        mirrors = [
            "https://ghproxy.net/" + url,
            "https://mirror.ghproxy.com/" + url,
            url
        ]

        success = False
        for mirror in mirrors:
            try:
                print(f"  尝试从镜像下载: {mirror[:60]}...")
                download_file_atomic(mirror, sam_file, expected_sha256=MOBILE_SAM_SHA256)
                print(f"\n  ✓ 下载完成: {sam_file}")
                downloaded_models += 1
                success = True
                break
            except Exception as e:
                print(f"\n  镜像失败: {e}")
                continue

        if not success:
            failed_models.append(("MobileSAM权重", "所有镜像都失败"))
except Exception as e:
    print(f"  ✗ 下载失败: {e}")
    failed_models.append(("MobileSAM权重", str(e)))

# 4. 下载MobileSAM源码
print("\n安装MobileSAM源码...")
try:
    import requests

    src_dir = PROJECT_ROOT / "src"
    target_dir = src_dir / "mobile_sam"

    if target_dir.exists():
        print("  ✓ 源码已存在,跳过")
    else:
        zip_url = "https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip"
        mirrors = [
            "https://ghproxy.net/" + zip_url,
            "https://mirror.ghproxy.com/" + zip_url,
            zip_url
        ]

        temp_zip = PROJECT_ROOT / "mobilesam.zip"
        temp_extract = PROJECT_ROOT / "mobilesam_temp"

        success = False
        for mirror in mirrors:
            try:
                print(f"  尝试从镜像下载: {mirror[:50]}...")
                response = requests.get(mirror, timeout=60)
                response.raise_for_status()

                with open(temp_zip, 'wb') as f:
                    f.write(response.content)

                # 解压
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)
                temp_extract.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(temp_zip, temp_extract)

                # 移动文件
                root_dir = next(temp_extract.glob("MobileSAM-*"), None)
                if root_dir and (root_dir / "mobile_sam").exists():
                    shutil.copytree(root_dir / "mobile_sam", target_dir)
                    print(f"  ✓ 源码安装完成: {target_dir}")
                    success = True
                    break
            except Exception as e:
                print(f"\n  镜像失败: {e}")
                continue
            finally:
                if temp_zip.exists():
                    temp_zip.unlink()
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)

        if not success:
            print("  ✗ 所有镜像都失败")
except Exception as e:
    print(f"  ✗ 安装失败: {e}")

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

# 6. 下载SAM2高精度分割模型 (可选)
print("\n" + "=" * 80)
print("[6/6] 下载SAM2高精度分割模型 (约155MB, 可选)...")
print("=" * 80)
print("提示: SAM2精度比MobileSAM高15-20%，但速度略慢")
user_input = prompt_input("是否下载? (y/n, 默认y): ").strip().lower()

if user_input != 'n':
    try:
        # 暂时禁用HF镜像，使用官方源
        original_endpoint = os.environ.pop('HF_ENDPOINT', None)

        from huggingface_hub import snapshot_download
        sam2_dir = MODELS_DIR / "sam2-hiera-tiny"

        if sam2_dir.exists():
            print("  ✓ 模型已存在,跳过")
            downloaded_models += 1
        else:
            print("  正在从HuggingFace官方下载...")
            snapshot_download(
                repo_id="facebook/sam2-hiera-tiny",
                local_dir=str(sam2_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4
            )
            print(f"  ✓ 下载完成: {sam2_dir}")
            downloaded_models += 1

        # 恢复镜像设置
        if original_endpoint:
            os.environ['HF_ENDPOINT'] = original_endpoint

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  提示: SAM2是可选项，MobileSAM已经足够使用")
        failed_models.append(("SAM2高精度模型", str(e)))
        # 恢复镜像设置
        if original_endpoint:
            os.environ['HF_ENDPOINT'] = original_endpoint
else:
    print("  ⊘ 跳过 SAM2模型下载")

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
