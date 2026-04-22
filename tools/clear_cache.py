#!/usr/bin/env python3
"""
清理 project-drcformer 目录中的所有 __pycache__ 缓存文件
"""
import os
import shutil
from pathlib import Path


def clear_pycache(root_dir):
    """
    递归清理指定目录下的所有 __pycache__ 目录
    
    Args:
        root_dir: 要清理的根目录路径
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"错误: 目录不存在: {root_dir}")
        return
    
    # 查找所有 __pycache__ 目录
    pycache_dirs = []
    for pycache_dir in root_path.rglob("__pycache__"):
        if pycache_dir.is_dir():
            pycache_dirs.append(pycache_dir)
    
    if not pycache_dirs:
        print(f"未找到任何 __pycache__ 目录")
        return
    
    # 删除所有找到的 __pycache__ 目录
    print(f"找到 {len(pycache_dirs)} 个 __pycache__ 目录:")
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"  ✓ 已删除: {pycache_dir}")
        except Exception as e:
            print(f"  ✗ 删除失败: {pycache_dir} - {e}")
    
    print(f"\n清理完成，共删除 {len(pycache_dirs)} 个 __pycache__ 目录")


if __name__ == "__main__":
    # 获取脚本所在目录作为根目录
    script_dir = Path(__file__).parent.absolute()
    clear_pycache(script_dir)

