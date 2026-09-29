#!/usr/bin/env python3
"""
检查 RKLLM 微调与部署环境所需的全部 Python 库
"""

import sys
import warnings

# 尝试用 pkg_resources，如果不可用就降级成 importlib.metadata
try:
    from pkg_resources import get_distribution, DistributionNotFound
    HAS_PKG_RESOURCES = True
except ImportError:
    try:
        from importlib.metadata import version as get_version, PackageNotFoundError as DistributionNotFound
        get_distribution = None
        HAS_PKG_RESOURCES = False
    except ImportError:
        print("❌ 无法导入 pkg_resources 或 importlib.metadata，请安装 setuptools。")
        sys.exit(1)

def get_pkg_version(pkg_name):
    """获取已安装包的版本号，失败返回 None"""
    if HAS_PKG_RESOURCES:
        try:
            return get_distribution(pkg_name).version
        except DistributionNotFound:
            pass
    else:
        try:
            return get_version(pkg_name)
        except DistributionNotFound:
            pass
    
    # 回退：尝试 import 并读取 __version__
    try:
        mod = __import__(pkg_name)
        return getattr(mod, '__version__', None)
    except ImportError:
        return None

# 需要检查的关键库（可自行增删）
REQUIRED_PACKAGES = [
    ("rkllm", "1.2.2"),          # RKLLM Toolkit 核心
    ("torch", None),              # PyTorch，不限定版本
    ("transformers", "4.40.0"),   # 建议 >= 4.40.0
    ("numpy", None),
    ("peft", "0.10.0"),           # LoRA 微调
    ("accelerate", "0.28.0"),
    ("datasets", "2.18.0"),
    ("huggingface_hub", "0.20.0"),
    ("sentencepiece", None),
    ("tokenizers", "0.19.0"),
    ("modelscope", None),         # 若使用魔搭环境
    ("bitsandbytes", "0.43.0"),   # 若开启量化训练
    ("uvicorn", None),            # 若部署 Web 服务
]

def compare_versions(version, target):
    """比较 version 是否 >= target (若 target 不为 None)"""
    if target is None:
        return True
    if version is None:
        return False
    parts_ver = [int(x) for x in str(version).split('.')]
    parts_tgt = [int(x) for x in target.split('.')]
    return parts_ver >= parts_tgt

def check_all():
    print("=" * 60)
    print("🐍 Python 版本:", sys.version.split()[0])
    print("=" * 60)
    all_ok = True
    for name, target in REQUIRED_PACKAGES:
        ver = get_pkg_version(name)
        if ver:
            status = "✅"
            if target and not compare_versions(ver, target):
                status = f"⚠️ (需要 >= {target}，实际 {ver})"
                all_ok = False
            print(f"{status} {name:20s}  {ver}")
        else:
            print(f"❌ {name:20s}  未安装")
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("🎉 所有关键库已就绪，可以继续下一步。")
    else:
        print("⚠️  缺少部分依赖，请按照上方提示安装。")
    print("=" * 60)

if __name__ == "__main__":
    check_all()