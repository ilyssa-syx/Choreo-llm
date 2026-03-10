#!/usr/bin/env python3
"""
download_all.py
===============
从 all.jsonl 中读取所有视频链接，使用 yt-dlp 并发下载到本地。

用法：
  python download_all.py
  python download_all.py --out_dir data/videos --workers 4
  python download_all.py --cookies cookies.txt --limit_rate 2M
  python download_all.py --resume          # 跳过 downloaded_path 已有的视频
  python download_all.py --update_jsonl    # 下载完写回 jsonl

选项：
  --jsonl       输入 jsonl 文件（默认 data/bili_solo_candidates/all.jsonl）
  --out_dir     下载根目录（默认 data/bili_solo_candidates/videos）
  --workers     并发线程数（默认 3）
  --cookies     yt-dlp cookies 文件路径（Netscape 格式）
  --limit_rate  yt-dlp 限速，例如 2M、500K（默认不限速）
  --format      yt-dlp 格式选择器（默认 bestvideo[height<=1080]+bestaudio/best）
  --resume      跳过 downloaded_path 字段已有本地文件的记录
  --update_jsonl 下载完成后将 downloaded_path 写回 jsonl 文件
  --timeout     单个视频下载超时秒数（默认 600）
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────── yt-dlp 是否可用 ───────────────────────
import shutil
if not shutil.which("yt-dlp"):
    logger.error("未找到 yt-dlp，请先安装：pip install yt-dlp")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ══════════════════════════════════════════════════════════════════
#  核心下载逻辑
# ══════════════════════════════════════════════════════════════════

_write_lock = Lock()


def build_cmd(url: str, out_path: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", args.format,
        "--output", str(out_path),
        "--no-warnings",
        "--retries", "3",
        "--fragment-retries", "3",
        "--merge-output-format", "mp4",
    ]
    if args.cookies and Path(args.cookies).exists():
        cmd += ["--cookies", args.cookies]
    if args.limit_rate:
        cmd += ["--limit-rate", args.limit_rate]
    cmd.append(url)
    return cmd


def download_one(rec: dict, out_dir: Path, args: argparse.Namespace) -> tuple[dict, bool]:
    """下载单条记录，返回 (更新后的 rec, 是否成功)。"""
    bvid = rec.get("bvid", "")
    url  = rec.get("url") or f"https://www.bilibili.com/video/{bvid}"
    out_path = out_dir / f"{bvid}.mp4"

    # 已有文件直接标记跳过
    if out_path.exists():
        rec["downloaded_path"] = str(out_path)
        logger.debug("已存在，跳过: %s", out_path)
        return rec, True

    # resume 模式：downloaded_path 有效时跳过
    if args.resume and rec.get("downloaded_path"):
        p = Path(rec["downloaded_path"])
        if p.exists():
            logger.debug("resume 跳过: %s", p)
            return rec, True

    cmd = build_cmd(url, out_path, args)
    logger.debug("cmd: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("超时 bvid=%s", bvid)
        return rec, False
    except Exception as exc:
        logger.error("异常 bvid=%s: %s", bvid, exc)
        return rec, False

    if result.returncode != 0:
        err_tail = result.stderr[-300:].strip()
        logger.warning("失败 bvid=%s (code=%d):\n  %s", bvid, result.returncode, err_tail)
        return rec, False

    # yt-dlp 可能以不同扩展名落盘
    if out_path.exists():
        rec["downloaded_path"] = str(out_path)
        return rec, True
    candidates = list(out_dir.glob(f"{bvid}.*"))
    if candidates:
        rec["downloaded_path"] = str(candidates[0])
        return rec, True

    logger.warning("退出码 0 但找不到文件 bvid=%s", bvid)
    return rec, False


# ══════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="从 all.jsonl 批量下载 B 站视频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jsonl", default="data/western_dance_candidates/all.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--out_dir", default="data/western_dance_candidates/videos",
        help="视频输出目录",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="并发下载线程数",
    )
    parser.add_argument(
        "--cookies", default=None,
        help="yt-dlp cookies 文件（Netscape 格式）",
    )
    parser.add_argument(
        "--limit_rate", default=None,
        help="yt-dlp 限速，例如 2M、500K",
    )
    parser.add_argument(
        "--format",
        default="bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best",
        help="yt-dlp 视频格式选择器",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="跳过 downloaded_path 已有本地文件的记录",
    )
    parser.add_argument(
        "--update_jsonl", action="store_true",
        help="下载完成后将 downloaded_path 写回原 jsonl 文件",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="单个视频下载超时秒数",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        logger.error("找不到 jsonl 文件: %s", jsonl_path)
        sys.exit(1)

    # 读取所有记录
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("共读取 %d 条记录，输出目录: %s", len(records), out_dir)

    # 统计
    ok_count   = 0
    fail_count = 0
    skip_count = 0
    updated_records = list(records)  # 保持顺序

    # 预先标记已存在的文件（快速 skip）
    to_download_idx: list[int] = []
    for i, rec in enumerate(records):
        bvid = rec.get("bvid", "")
        out_path = out_dir / f"{bvid}.mp4"
        if out_path.exists():
            updated_records[i]["downloaded_path"] = str(out_path)
            skip_count += 1
            ok_count   += 1
        elif args.resume and rec.get("downloaded_path") and Path(rec["downloaded_path"]).exists():
            skip_count += 1
            ok_count   += 1
        else:
            to_download_idx.append(i)

    logger.info(
        "预扫描: %d 个已存在（跳过），%d 个待下载",
        skip_count, len(to_download_idx),
    )

    if not to_download_idx:
        logger.info("全部已下载，无需操作。")
    else:
        # 并发下载
        t0 = time.time()
        iter_ctx = tqdm(total=len(to_download_idx), desc="Downloading", unit="video") if HAS_TQDM else None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(download_one, records[i], out_dir, args): i
                for i in to_download_idx
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    updated_rec, success = future.result()
                    updated_records[idx] = updated_rec
                    if success:
                        ok_count += 1
                    else:
                        fail_count += 1
                except Exception as exc:
                    logger.error("Future 异常 idx=%d: %s", idx, exc)
                    fail_count += 1
                finally:
                    if iter_ctx:
                        iter_ctx.set_postfix(ok=ok_count, fail=fail_count)
                        iter_ctx.update(1)

        if iter_ctx:
            iter_ctx.close()

        elapsed = time.time() - t0
        logger.info(
            "完成：成功 %d / 失败 %d / 跳过 %d，耗时 %.1f 分钟",
            ok_count - skip_count, fail_count, skip_count, elapsed / 60,
        )

    # 写回 jsonl
    if args.update_jsonl:
        tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in updated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(jsonl_path)
        logger.info("已更新 downloaded_path 写回 %s", jsonl_path)


if __name__ == "__main__":
    main()
