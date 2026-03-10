#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
from pathlib import Path

def run(cmd):
    # 让你能看到错误信息
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def convert_one(in_path: Path, out_path: Path, overwrite: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and (not overwrite):
        print(f"[SKIP] exists: {out_path}")
        return True

    # 关键点：
    # -vf fps=30 + -vsync cfr：强制恒定 30fps（对 VFR 也更稳）
    # -c:v libx264：重编码视频（帧率转换必然需要重编码）
    # -c:a copy：音频直接拷贝（不重编码）
    # 如你希望音频也重编码（更保险），把 -c:a copy 改成 -c:a aac -b:a 192k
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(in_path),
        "-map", "0",
        "-vf", "fps=30",
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(out_path),
    ]

    p = run(cmd)
    if p.returncode != 0:
        print(f"[FAIL] {in_path}\n  -> {out_path}\n{p.stderr.strip()}")
        return False

    print(f"[OK] {in_path}\n  -> {out_path}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="输入根目录（递归找 .mp4）")
    ap.add_argument("--output_dir", required=True, help="输出根目录（保持子目录结构）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在输出文件")
    args = ap.parse_args()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    # 检查 ffmpeg
    if run(["ffmpeg", "-version"]).returncode != 0:
        raise SystemExit("找不到 ffmpeg，请先安装或确保 ffmpeg 在 PATH 中。")

    mp4s = sorted(in_root.rglob("*.mp4"))
    print(f"Found {len(mp4s)} mp4 files under: {in_root}")

    ok, fail = 0, 0
    for in_path in mp4s:
        rel = in_path.relative_to(in_root)  # 保持目录结构
        out_path = out_root / rel
        # 你也可以选择改名，比如加 _30fps 后缀：
        # out_path = (out_root / rel).with_name(in_path.stem + "_30fps.mp4")

        if convert_one(in_path, out_path, args.overwrite):
            ok += 1
        else:
            fail += 1

    print("=" * 60)
    print(f"Done. OK: {ok}, FAIL: {fail}")
    print(f"Output root: {out_root}")

if __name__ == "__main__":
    main()