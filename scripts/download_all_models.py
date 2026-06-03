#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一模型下载脚本 - 完整版（包含所有必需模型）
支持的模型：
1. NLP理解模型 (paraphrase-multilingual-MiniLM-L12-v2)
2. 深度估计模型 (depth-anything-small)
3. MobileSAM 分割模型
4. CLIP 多语言图像检索模型
5. Qwen 大语言模型 (可选)
6. SAM2 高精度分割模型 (可选)
"""
import os
import sys
import hashlib
import shutil
import zipfile
from pathlib import Path

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

MOBILE_SAM_SHA256 = "f3c0d8cda613564d499310dab6c812cd80d9de20dd0e7d7b3ea0cd86ff5c76d6"


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

print("=" * 80)
print("   AI影像处理软件 - 完整模型下载工具")
print("=" * 80)
print("\n📦 将下载以下模型:")
print("  [必需] 1. NLP理解模型 (~471MB)")
print("  [必需] 2. 深度估计模型 (~99MB)")
print("  [必需] 3. MobileSAM分割模型 (~40MB)")
print("  [必需] 4. CLIP多语言模型 (~540MB)")
print("  [可选] 5. Qwen大语言模型 (~3GB)")
print("  [可选] 6. SAM2高精度分割 (~155MB)")
print("\n⏱️  预计总下载时间: 10-30分钟 (取决于网络速度)")
print("=" * 80)

# 下载进度统计
total_models = 6
downloaded_models = 0
failed_models = []

# 1. 下载NLP理解模型
print("\n" + "=" * 80)
print("[1/6] 下载NLP理解模型 (约471MB)...")
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

# 2. 下载深度估计模型
print("\n" + "=" * 80)
print("[2/6] 下载深度估计模型 (约99MB)...")
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

# 3. 下载MobileSAM权重
print("\n" + "=" * 80)
print("[3/6] 下载MobileSAM权重 (约40MB)...")
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

# 4. 下载多语言CLIP模型
print("\n" + "=" * 80)
print("[4/6] 下载多语言CLIP模型 (约540MB)...")
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

# 5. 下载Qwen大语言模型 (可选)
print("\n" + "=" * 80)
print("[5/6] 下载Qwen2.5-1.5B-Instruct大语言模型 (约3GB, 可选)...")
print("=" * 80)
print("提示: 此模型较大且为可选项，如不需要AI语义分析可跳过")
user_input = input("是否下载? (y/n, 默认n): ").strip().lower()

if user_input == 'y':
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        save_path = MODELS_DIR / "Qwen2.5-1.5B-Instruct"
        if save_path.exists():
            print("  ✓ 模型已存在,跳过")
            downloaded_models += 1
        else:
            print("  正在下载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            tokenizer.save_pretrained(str(save_path))

            print("  正在下载模型权重 (约3GB,可能需要较长时间)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="cpu",
                trust_remote_code=False
            )
            model.save_pretrained(str(save_path))
            print(f"  ✓ 下载完成: {save_path}")
            downloaded_models += 1
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        print("  提示: 如果网络问题导致失败,可以稍后重试")
        failed_models.append(("Qwen大语言模型", str(e)))
else:
    print("  ⊘ 跳过 Qwen模型下载")

# 6. 下载SAM2高精度分割模型 (可选)
print("\n" + "=" * 80)
print("[6/6] 下载SAM2高精度分割模型 (约155MB, 可选)...")
print("=" * 80)
print("提示: SAM2精度比MobileSAM高15-20%，但速度略慢")
user_input = input("是否下载? (y/n, 默认y): ").strip().lower()

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
print("  3. 大语言模型(Qwen)需在 llm_config.json 中启用")
print("  4. 使用 src/utils/model_checksum.py 验证模型完整性")
print("\n" + "=" * 80)
