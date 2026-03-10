#!/usr/bin/env python3
"""
Select top videos by weighted score until reaching target hours.

This script reads a JSONL (e.g., all.jsonl) with per-video scores,
filters to videos that exist in a given folder, computes the weighted
score, sorts descending, and selects videos until the accumulated
runtime meets or exceeds the target.

It does not delete anything by default. You can optionally move
non-selected videos to another folder for cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_WEIGHTS = {
    "weight_solo": 0.55,
    "weight_audio": 0.30,
    "weight_metadata": 0.15,
}

VIDEO_EXTS = (".mp4", ".flv", ".webm", ".mkv", ".avi")


@dataclass
class ScoredVideo:
    bvid: str
    path: Path
    duration_sec: float
    score: float
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top videos by weighted score until target hours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Input JSONL file (e.g., data/.../all.jsonl)",
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Directory containing downloaded videos",
    )
    parser.add_argument(
        "--target_hours",
        type=float,
        required=True,
        help="Target hours to keep (stop when reached/exceeded)",
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=300.0,
        help="Exclude videos longer than this many seconds",
    )
    parser.add_argument(
        "--include_likely_multi",
        action="store_true",
        default=False,
        help="Include videos with solo_label == 'likely_multi'",
    )
    parser.add_argument(
        "--out_jsonl",
        default="selected.jsonl",
        help="Output JSONL for selected items",
    )
    parser.add_argument(
        "--out_list",
        default="selected_paths.txt",
        help="Output text file with selected video paths",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Override weights as JSON string, e.g. "
            "'{'\"weight_solo\":0.6,'\"weight_audio\":0.25,'\"weight_metadata\":0.15}'"
        ),
    )
    parser.add_argument(
        "--move_nonselected_dir",
        default=None,
        help="If set, move non-selected videos into this directory",
    )
    parser.add_argument(
        "--probe_missing_duration",
        action="store_true",
        help="Use ffprobe to fill missing duration_sec",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_video_file(video_dir: Path, bvid: str) -> Path | None:
    for ext in VIDEO_EXTS:
        candidate = video_dir / f"{bvid}{ext}"
        if candidate.exists():
            return candidate
    return None


def ffprobe_duration(video_path: Path) -> float | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        value = result.stdout.strip()
        return float(value) if value else None
    except Exception:
        return None


def compute_score(rec: dict[str, Any], weights: dict[str, float]) -> float | None:
    if not rec.get("metadata_filter_pass", True):
        return None

    w_solo = weights["weight_solo"]
    w_audio = weights["weight_audio"]
    w_meta = weights["weight_metadata"]
    total_w = w_solo + w_audio + w_meta

    solo_score = rec.get("solo_score", 0.5)
    audio_score = rec.get("audio_score", 0.5)
    meta_score = rec.get("metadata_score", 0.0)

    if not rec.get("vision_checked", False):
        solo_score = 0.5
    if not rec.get("audio_checked", False):
        audio_score = 0.5

    raw = (solo_score * w_solo + audio_score * w_audio + meta_score * w_meta) / total_w
    return float(raw)


def collect_scored_videos(
    jsonl_path: Path,
    video_dir: Path,
    weights: dict[str, float],
    probe_missing_duration: bool,
    max_duration_sec: float,
    include_likely_multi: bool,
) -> list[ScoredVideo]:
    results: list[ScoredVideo] = []
    for rec in load_jsonl(jsonl_path):
        bvid = rec.get("bvid", "")
        if not bvid:
            continue

        if not include_likely_multi and rec.get("solo_label") == "likely_multi":
            continue

        video_path = find_video_file(video_dir, bvid)
        if video_path is None:
            continue

        score = compute_score(rec, weights)
        if score is None:
            continue

        duration_sec = rec.get("duration_sec")
        if duration_sec is None and probe_missing_duration:
            duration_sec = ffprobe_duration(video_path)

        if duration_sec is None:
            continue

        if float(duration_sec) > max_duration_sec:
            continue

        results.append(
            ScoredVideo(
                bvid=bvid,
                path=video_path,
                duration_sec=float(duration_sec),
                score=score,
                record=rec,
            )
        )
    return results


def select_top_hours(scored: list[ScoredVideo], target_hours: float) -> list[ScoredVideo]:
    target_sec = target_hours * 3600.0
    scored_sorted = sorted(scored, key=lambda s: s.score, reverse=True)
    selected: list[ScoredVideo] = []
    total_sec = 0.0

    for item in scored_sorted:
        selected.append(item)
        total_sec += item.duration_sec
        if total_sec >= target_sec:
            break

    return selected


def write_outputs(selected: list[ScoredVideo], out_jsonl: Path, out_list: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in selected:
            rec = dict(item.record)
            rec["final_score"] = round(item.score, 6)
            rec["selected_path"] = str(item.path)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_list, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(str(item.path) + "\n")


def move_nonselected(
    all_items: list[ScoredVideo],
    selected: list[ScoredVideo],
    move_dir: Path,
) -> None:
    move_dir.mkdir(parents=True, exist_ok=True)
    selected_paths = {item.path for item in selected}
    for item in all_items:
        if item.path in selected_paths:
            continue
        target = move_dir / item.path.name
        if target.exists():
            continue
        shutil.move(str(item.path), str(target))


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl)
    video_dir = Path(args.video_dir)
    out_jsonl = Path(args.out_jsonl)
    out_list = Path(args.out_list)

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))

    scored = collect_scored_videos(
        jsonl_path=jsonl_path,
        video_dir=video_dir,
        weights=weights,
        probe_missing_duration=args.probe_missing_duration,
        max_duration_sec=args.max_duration_sec,
        include_likely_multi=args.include_likely_multi,
    )

    selected = select_top_hours(scored, args.target_hours)
    write_outputs(selected, out_jsonl, out_list)

    if args.move_nonselected_dir:
        move_nonselected(scored, selected, Path(args.move_nonselected_dir))

    total_sec = sum(s.duration_sec for s in selected)
    print(f"Selected {len(selected)} videos, total_hours={total_sec/3600:.2f}")


if __name__ == "__main__":
    main()
