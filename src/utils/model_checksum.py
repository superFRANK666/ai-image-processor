"""
模型下载工具 - 带SHA256完整性校验
提供安全的模型文件下载和验证功能
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """
    计算文件的SHA256哈希值

    Args:
        file_path: 文件路径
        chunk_size: 读取块大小

    Returns:
        SHA256哈希值（十六进制字符串）
    """
    sha256_hash = hashlib.sha256()
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"校验 {file_path.name}") as pbar:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256_hash.update(data)
                pbar.update(len(data))

    return sha256_hash.hexdigest()


def verify_file_integrity(file_path: Path, expected_sha256: str) -> bool:
    """
    验证文件完整性

    Args:
        file_path: 文件路径
        expected_sha256: 预期的SHA256哈希值

    Returns:
        True if 文件完整, False otherwise
    """
    if not file_path.exists():
        print(f"✗ 文件不存在: {file_path}")
        return False

    print(f"\n正在验证文件完整性: {file_path.name}")
    actual_sha256 = calculate_sha256(file_path)

    if actual_sha256.lower() == expected_sha256.lower():
        print(f"✓ 文件完整性验证通过")
        print(f"  SHA256: {actual_sha256}")
        return True
    else:
        print(f"✗ 文件完整性验证失败!")
        print(f"  预期: {expected_sha256}")
        print(f"  实际: {actual_sha256}")
        return False


def save_checksums(model_dir: Path, checksums: Dict[str, str]):
    """
    保存校验和到文件

    Args:
        model_dir: 模型目录
        checksums: 文件名到SHA256的映射
    """
    checksum_file = model_dir / "SHA256SUMS.json"
    with open(checksum_file, 'w', encoding='utf-8') as f:
        json.dump(checksums, f, indent=2)
    print(f"✓ 校验和已保存到: {checksum_file}")


def load_checksums(model_dir: Path) -> Optional[Dict[str, str]]:
    """
    从文件加载校验和

    Args:
        model_dir: 模型目录

    Returns:
        文件名到SHA256的映射，如果文件不存在则返回None
    """
    checksum_file = model_dir / "SHA256SUMS.json"
    if not checksum_file.exists():
        return None

    try:
        with open(checksum_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ 读取校验和文件失败: {e}")
        return None


def verify_model_integrity(model_dir: Path, critical_files: Optional[list] = None) -> bool:
    """
    验证模型目录中所有文件的完整性

    Args:
        model_dir: 模型目录
        critical_files: 关键文件列表（必须存在且验证通过），如果为None则验证所有文件

    Returns:
        True if 所有文件完整, False otherwise
    """
    checksums = load_checksums(model_dir)

    if checksums is None:
        print(f"⚠ 未找到校验和文件，无法验证完整性")
        print(f"  提示: 首次下载后会自动生成校验和")
        return True  # 首次下载没有校验和，默认通过

    print(f"\n开始验证模型完整性: {model_dir.name}")

    all_valid = True
    verified_count = 0

    for file_name, expected_sha256 in checksums.items():
        file_path = model_dir / file_name

        # 如果指定了关键文件列表，只验证关键文件
        if critical_files is not None and file_name not in critical_files:
            continue

        if not file_path.exists():
            print(f"✗ 文件缺失: {file_name}")
            all_valid = False
            continue

        if verify_file_integrity(file_path, expected_sha256):
            verified_count += 1
        else:
            all_valid = False
            print(f"⚠ 建议重新下载: {file_name}")

    if all_valid:
        print(f"\n✓ 模型完整性验证通过 ({verified_count}/{len(checksums)} 文件)")
    else:
        print(f"\n✗ 模型完整性验证失败")
        print(f"  建议操作: 删除模型目录并重新下载")

    return all_valid


def generate_checksums(model_dir: Path, file_patterns: Optional[list] = None):
    """
    为模型目录生成校验和

    Args:
        model_dir: 模型目录
        file_patterns: 文件模式列表（如 ['*.pt', '*.bin', '*.safetensors']）
                      如果为None，则处理所有文件
    """
    if not model_dir.exists():
        print(f"✗ 目录不存在: {model_dir}")
        return

    print(f"\n为模型生成校验和: {model_dir.name}")

    # 收集文件
    files_to_check = []
    if file_patterns:
        for pattern in file_patterns:
            files_to_check.extend(model_dir.glob(pattern))
    else:
        files_to_check = [f for f in model_dir.rglob("*") if f.is_file()]

    # 排除校验和文件本身
    files_to_check = [f for f in files_to_check if f.name != "SHA256SUMS.json"]

    if not files_to_check:
        print(f"⚠ 未找到要校验的文件")
        return

    checksums = {}
    for file_path in files_to_check:
        relative_path = file_path.relative_to(model_dir)
        sha256 = calculate_sha256(file_path)
        checksums[str(relative_path).replace('\\', '/')] = sha256
        print(f"  {relative_path.name}: {sha256}")

    save_checksums(model_dir, checksums)
    print(f"\n✓ 已生成 {len(checksums)} 个文件的校验和")


# 已知模型的SHA256校验和（由社区维护）
KNOWN_MODEL_CHECKSUMS = {
    "mobile_sam.pt": {
        "sha256": "f3c0d8cda613564d499310dab6c812cd80d9de20dd0e7d7b3ea0cd86ff5c76d6",
        "size_mb": 39.0,
        "source": "官方发布"
    },
    # 可以添加更多已知模型的校验和
}


def check_against_known_checksums(file_path: Path) -> Optional[bool]:
    """
    对照已知校验和检查文件

    Args:
        file_path: 文件路径

    Returns:
        True if 匹配, False if 不匹配, None if 未知
    """
    file_name = file_path.name

    if file_name not in KNOWN_MODEL_CHECKSUMS:
        return None  # 未知文件

    known_info = KNOWN_MODEL_CHECKSUMS[file_name]
    expected_sha256 = known_info["sha256"]

    print(f"\n检查已知模型: {file_name}")
    print(f"  来源: {known_info['source']}")
    print(f"  预期大小: ~{known_info['size_mb']:.1f}MB")

    return verify_file_integrity(file_path, expected_sha256)


if __name__ == "__main__":
    # 示例用法
    from pathlib import Path
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  生成校验和: python model_checksum.py generate <model_dir>")
        print("  验证模型:   python model_checksum.py verify <model_dir>")
        print("  检查文件:   python model_checksum.py check <file_path>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        if len(sys.argv) < 3:
            print("请指定模型目录")
            sys.exit(1)
        model_dir = Path(sys.argv[2])
        generate_checksums(model_dir, ['*.pt', '*.bin', '*.safetensors', '*.onnx'])

    elif command == "verify":
        if len(sys.argv) < 3:
            print("请指定模型目录")
            sys.exit(1)
        model_dir = Path(sys.argv[2])
        if verify_model_integrity(model_dir):
            print("\n✓ 验证通过")
            sys.exit(0)
        else:
            print("\n✗ 验证失败")
            sys.exit(1)

    elif command == "check":
        if len(sys.argv) < 3:
            print("请指定文件路径")
            sys.exit(1)
        file_path = Path(sys.argv[2])
        result = check_against_known_checksums(file_path)
        if result is True:
            print("\n✓ 文件完整")
            sys.exit(0)
        elif result is False:
            print("\n✗ 文件损坏")
            sys.exit(1)
        else:
            print("\n⚠ 未知文件（无已知校验和）")
            # 生成校验和供参考
            sha256 = calculate_sha256(file_path)
            print(f"  SHA256: {sha256}")
            sys.exit(0)

    else:
        print(f"未知命令: {command}")
        sys.exit(1)
