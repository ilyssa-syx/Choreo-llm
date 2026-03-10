#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import copy
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def merge_segments(
    captions: List[Tuple[int, List[Dict[str, Any]]]],
    fps: int = 30,
    step_seconds: float = 0.5,
    min_seg_len: int = 1,
) -> List[Dict[str, Any]]:
    """
    将多个切片视频的 segments 合并成完整时间线（全局帧）
    - captions: list of (slice_idx, segments); 支持不连续的 slice
    - step_seconds: 相邻 slice 的时间偏移（秒）
    """
    step_frames = int(step_seconds * fps)

    # Step 1: 提取所有segment并映射到全局帧数
    all_segments: List[Dict[str, Any]] = []
    for slice_idx, caption in captions:
        caption_start_frame = slice_idx * step_frames
        for segment in caption:
            if "start_frame" not in segment or "end_frame" not in segment:
                continue
            if int(segment["end_frame"]) - int(segment["start_frame"]) < min_seg_len:
                continue

            global_segment = copy.deepcopy(segment)
            global_segment["start_frame"] = int(segment["start_frame"]) + caption_start_frame
            global_segment["end_frame"] = int(segment["end_frame"]) + caption_start_frame
            if "key_frame" in segment and isinstance(segment["key_frame"], list):
                global_segment["key_frame"] = [int(kf) + caption_start_frame for kf in segment["key_frame"]]
            else:
                global_segment["key_frame"] = []
            global_segment["slice_idx"] = slice_idx
            global_segment["original_start"] = int(segment["start_frame"])
            global_segment["original_end"] = int(segment["end_frame"])
            all_segments.append(global_segment)

    if not all_segments:
        return []

    # Step 2: 建立 start_frame -> segments 的字典
    frame_to_segments: Dict[int, List[Dict[str, Any]]] = {}
    for seg in all_segments:
        s = int(seg["start_frame"])
        frame_to_segments.setdefault(s, []).append(seg)

    # Step 3: 全局结束帧
    total_end_frame = max(int(seg["end_frame"]) for seg in all_segments)

    # Step 4: 主循环合并
    result: List[Dict[str, Any]] = []
    cur_frame = 0

    # 为了更快：预排序 segments，便于找“下一个”
    all_segments_sorted = sorted(all_segments, key=lambda x: (int(x["start_frame"]), int(x["end_frame"])))

    guard = 0
    max_guard = total_end_frame + 10000  # 防止极端情况下无限循环

    while cur_frame < total_end_frame:
        guard += 1
        if guard > max_guard:
            # 极端兜底：直接跳出
            break

        prev_frame = cur_frame

        # 1) 正好有 start_frame==cur_frame 的候选
        if cur_frame in frame_to_segments:
            candidates = frame_to_segments[cur_frame]
            selected = random.choice(candidates)
            result.append(selected)
            cur_frame = int(selected["end_frame"])

        else:
            # 2) 找一个能覆盖/推进 cur_frame 的 segment：
            # 优先找 start<=cur<end 的（覆盖当前），否则找 start>cur 的最早那个（推进）
            cover = None
            next_after = None

            # 先尝试覆盖（线性扫，规模不大时够用；若很大可再优化）
            for seg in all_segments_sorted:
                s = int(seg["start_frame"])
                e = int(seg["end_frame"])
                if s <= cur_frame < e:
                    # 持续更新，选择s离cur_frame最近的（即s最大的）
                    cover = seg
                elif s > cur_frame:
                    if next_after is None:
                        next_after = seg
                    break  # 后面的s只会更大，不需要继续

            chosen = cover if cover is not None else next_after
            if chosen is None:
                # 真的找不到任何能推进的，兜底推进一帧
                cur_frame += 1
            else:
                app = chosen.copy()
                # 把 start 改成当前帧，让时间线连续
                app["start_frame"] = cur_frame
                result.append(app)
                cur_frame = int(app["end_frame"])

        # 兜底：如果没有推进，强制推进 1 帧，避免卡死
        if cur_frame <= prev_frame:
            cur_frame = prev_frame + 1

    # Step 5: 合并过短片段（避免在遍历时 remove 导致漏检）
    if len(result) > 1:
        cleaned: List[Dict[str, Any]] = [result[0]]
        for seg in result[1:]:
            if int(seg["end_frame"]) - int(seg["start_frame"]) < 10:
                # 合并到前一个
                cleaned[-1]["end_frame"] = int(seg["end_frame"])
            else:
                cleaned.append(seg)
        result = cleaned

    return result


def group_slice_files_in_dir(dir_path: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """
    在单个目录内，把形如 xxx_sliceK.json 的文件按 base_name=xxx 分组
    返回: base_name -> [(slice_idx, path), ...]
    """
    groups: Dict[str, List[Tuple[int, Path]]] = {}
    for p in dir_path.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        name = p.name
        if "_slice" not in name:
            continue
        base, rest = name.split("_slice", 1)
        try:
            slice_idx = int(rest.split(".", 1)[0])
        except Exception:
            continue
        groups.setdefault(base, []).append((slice_idx, p))
    return groups


def process_tree(
    input_root: Path,
    output_root: Path,
    fps: int = 30,
    step_seconds: float = 0.5,
    wrap_output: bool = False,
) -> None:
    """
    递归处理 input_root 下所有子目录：
    - 每个子目录里按 xxx_sliceK.json 分组
    - 合并后写入 output_root/相对路径/xxx.json
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 遍历所有“包含 json 的目录”
    # 用 rglob 找 json，然后取 parent 去重
    parents = sorted({p.parent for p in input_root.rglob("*.json")})

    for in_dir in parents:
        groups = group_slice_files_in_dir(in_dir)
        if not groups:
            continue

        rel_dir = in_dir.relative_to(input_root)
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for base_name, items in groups.items():
            items.sort(key=lambda x: x[0])
            captions: List[Tuple[int, List[Dict[str, Any]]]] = []
            for slice_idx, path in items:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # 兼容：文件里可能是 dict 包着 segments
                if not isinstance(data, list):
                    raise ValueError(f"JSON format unexpected: {path}")
                captions.append((slice_idx, data))

            merged = merge_segments(captions, fps=fps, step_seconds=step_seconds)

            out_path = out_dir / f"{base_name}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=4)

            print(f"[OK] {in_dir} : merged {len(items)} -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True, help="输入根目录（递归）")
    ap.add_argument("--output_root", type=str, required=True, help="输出根目录（保持结构）")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--step_seconds", type=float, default=0.5, help="相邻 slice 偏移（秒），无 overlap 用 0.5")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_tree(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        fps=args.fps,
        step_seconds=args.step_seconds
    )