#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make side-by-side videos:
- Left: original mp4
- Right: white panel with black text (modifier) shown when frame_idx in [start_frame, end_frame)

Folder traversal:
- video_dir: recursively find *.mp4
- anno_dir: find corresponding JSON by same relative path and same stem (suffix .json)
- out_dir: write output mp4 keeping the same relative structure

Annotation JSON format (relevant fields):
{
  "merged_segments": [
    {"start_frame": 47, "end_frame": 88, "modifier": "..."},
    ...
  ],
  "trim_frame": 345
}

Notes:
- Assumes your video has been converted to 30fps already (as you said).
- Uses end_frame as exclusive to avoid overlaps when next segment starts at same frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class Segment:
    start: int
    end: int  # exclusive
    modifier: str


def load_segments(json_path: Path) -> Tuple[List[Segment], Optional[int]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segs = []
    for s in data:
        if "start_frame" not in s or "end_frame" not in s:
            continue
        start = int(s["start_frame"])
        end = int(s["end_frame"])
        modifier = str(s.get("description", "")).strip()

        # Use [start, end) to avoid overlap (your example has end==next_start)
        if end <= start:
            continue
        segs.append(Segment(start=start, end=end, modifier=modifier))

    segs.sort(key=lambda x: (x.start, x.end))
    return segs


def get_active_modifier(segs: List[Segment], frame_idx: int, cursor: int) -> Tuple[str, int]:
    """
    segs sorted by start.
    cursor is a moving pointer to avoid scanning from beginning each frame.
    Returns (modifier_text_or_empty, new_cursor).
    """
    n = len(segs)
    # Advance cursor if current frame is beyond current segment end
    while cursor < n and frame_idx >= segs[cursor].end:
        cursor += 1
    if cursor < n and segs[cursor].start <= frame_idx < segs[cursor].end:
        return segs[cursor].modifier, cursor
    return "", cursor


def wrap_text_to_width(
    text: str,
    font_face: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> List[str]:
    """
    Wrap text to fit max_width in pixels using cv2.getTextSize.
    Preserves existing newlines; wraps each line individually.
    """
    lines_out: List[str] = []
    raw_lines = text.splitlines() if text else [""]

    def text_width(s: str) -> int:
        (w, _), _ = cv2.getTextSize(s, font_face, font_scale, thickness)
        return w

    def break_long_token(token: str) -> List[str]:
        # If a single "word" is too long, break by characters.
        if token == "":
            return [token]
        chunks = []
        cur = ""
        for ch in token:
            candidate = cur + ch
            if cur and text_width(candidate) > max_width:
                chunks.append(cur)
                cur = ch
            else:
                cur = candidate
        if cur:
            chunks.append(cur)
        return chunks

    for raw in raw_lines:
        if raw.strip() == "":
            lines_out.append("")
            continue

        words = raw.split(" ")
        cur_line = ""
        for w in words:
            if w == "":
                # multiple spaces
                candidate = (cur_line + " ") if cur_line else ""
                if candidate and text_width(candidate) <= max_width:
                    cur_line = candidate
                continue

            # Break long tokens if needed
            parts = [w] if text_width(w) <= max_width else break_long_token(w)

            for part in parts:
                if cur_line == "":
                    if text_width(part) <= max_width:
                        cur_line = part
                    else:
                        # extreme edge case, still append
                        lines_out.append(part)
                        cur_line = ""
                    continue

                candidate = cur_line + " " + part
                if text_width(candidate) <= max_width:
                    cur_line = candidate
                else:
                    lines_out.append(cur_line)
                    cur_line = part

        if cur_line != "":
            lines_out.append(cur_line)

    return lines_out


def render_text_panel(
    height: int,
    width: int,
    text: str,
    font_face: int,
    font_scale: float,
    thickness: int,
    margin: int,
    line_spacing: int,
    header: Optional[str] = None,
) -> np.ndarray:
    """
    White background (uint8 255), black text.
    If text is empty, returns blank white panel (optionally with header).
    """
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    x = margin
    y = margin

    # Header (optional)
    if header:
        header_lines = wrap_text_to_width(header, font_face, font_scale, thickness, width - 2 * margin)
        for hl in header_lines:
            (tw, th), baseline = cv2.getTextSize(hl, font_face, font_scale, thickness)
            y_line = y + th
            if y_line + baseline > height - margin:
                return panel
            cv2.putText(panel, hl, (x, y_line), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y = y_line + baseline + line_spacing
        y += line_spacing  # extra gap after header

    if not text:
        return panel

    max_w = width - 2 * margin
    wrapped = wrap_text_to_width(text, font_face, font_scale, thickness, max_w)

    # Compute line height
    (_, th), baseline = cv2.getTextSize("Ag", font_face, font_scale, thickness)
    line_h = th + baseline + line_spacing

    # Draw lines; truncate if too many
    max_lines = max(1, (height - margin - y) // max(1, line_h))
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        if max_lines >= 1:
            # Add ellipsis to last line if possible
            last = wrapped[-1]
            ell = " ..."
            # ensure last line with ellipsis fits
            while last and cv2.getTextSize(last + ell, font_face, font_scale, thickness)[0][0] > max_w:
                last = last[:-1]
            wrapped[-1] = (last + ell).strip()

    for line in wrapped:
        y_line = y + th
        if y_line + baseline > height - margin:
            break
        cv2.putText(panel, line, (x, y_line), font_face, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_h

    return panel


def ffmpeg_mux_audio(original_mp4: Path, video_noaudio_mp4: Path, out_mp4: Path) -> bool:
    """
    Try to mux audio from original_mp4 into video_noaudio_mp4.
    If original has no audio, ffmpeg will fail; then we return False.
    """
    if shutil.which("ffmpeg") is None:
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_noaudio_mp4),
        "-i", str(original_mp4),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out_mp4),
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return p.returncode == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0
    except Exception:
        return False


def process_one_video(
    video_path: Path,
    json_path: Path,
    out_path: Path,
    overwrite: bool,
    keep_audio: bool,
    font_scale: float,
    thickness: int,
    margin: int,
    line_spacing: int,
    fourcc: str,
    show_header: bool,
) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return True

    segs = load_segments(json_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output size: 2W x H
    out_w, out_h = 2 * w, h

    # Writer (silent)
    tmp_noaudio = out_path.with_suffix(".noaudio_tmp.mp4")
    if tmp_noaudio.exists():
        tmp_noaudio.unlink()

    writer = cv2.VideoWriter(
        str(tmp_noaudio),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        print(f"[WARN] Cannot open VideoWriter for: {tmp_noaudio}")
        cap.release()
        return False

    # Auto-tune font scale a bit by video height if user passed <=0
    if font_scale <= 0:
        # around 0.7 at 720p
        font_scale_use = max(0.45, min(1.2, 0.7 * (h / 720.0)))
    else:
        font_scale_use = font_scale

    font_face = cv2.FONT_HERSHEY_SIMPLEX

    cursor = 0
    frame_idx = 0

    # Determine max frames to process
    # trim_frame is number of frames (end cap), so process [0, trim_frame)
    max_frames = None
    

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        modifier, cursor = get_active_modifier(segs, frame_idx, cursor)

        header = None
        if show_header:
            header = f"frame: {frame_idx}"
        panel = render_text_panel(
            height=h,
            width=w,
            text=modifier,
            font_face=font_face,
            font_scale=font_scale_use,
            thickness=thickness,
            margin=margin,
            line_spacing=line_spacing,
            header=header,
        )

        combined = np.concatenate([frame, panel], axis=1)
        writer.write(combined)

        frame_idx += 1

    writer.release()
    cap.release()

    # Mux audio back if requested
    if keep_audio:
        if out_path.exists():
            out_path.unlink()
        mux_ok = ffmpeg_mux_audio(video_path, tmp_noaudio, out_path)
        if mux_ok:
            tmp_noaudio.unlink(missing_ok=True)
            return True
        # fallback: just move no-audio video
        tmp_noaudio.replace(out_path)
        return True
    else:
        tmp_noaudio.replace(out_path)
        return True


def iter_videos(video_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    vids = []
    for ext in exts:
        vids.extend(video_dir.rglob(f"*{ext}"))
    vids = [p for p in vids if p.is_file()]
    vids.sort()
    return vids


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create side-by-side mp4 with modifier text panel (recursive, keep folder structure)."
    )
    ap.add_argument("--video_dir", type=str, required=True, help="Root folder containing .mp4 videos (recursive).")
    ap.add_argument("--anno_dir", type=str, required=True, help="Root folder containing annotation .json files (same structure).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output root folder (structure preserved).")

    ap.add_argument("--video_ext", type=str, default=".mp4", help="Video extension to search.")
    ap.add_argument("--anno_ext", type=str, default=".json", help="Annotation extension (default .json).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--keep_audio", action="store_true", help="Try muxing original audio back using ffmpeg (recommended).")
    ap.add_argument("--no_keep_audio", action="store_true", help="Do not mux audio back (output will be silent).")

    ap.add_argument("--font_scale", type=float, default=0.0, help="cv2 font scale (<=0 means auto by height).")
    ap.add_argument("--thickness", type=int, default=2, help="cv2 putText thickness.")
    ap.add_argument("--margin", type=int, default=20, help="Text margin (pixels).")
    ap.add_argument("--line_spacing", type=int, default=8, help="Extra spacing between lines (pixels).")

    ap.add_argument("--fourcc", type=str, default="mp4v", help="FourCC for output video (mp4v usually works).")
    ap.add_argument("--show_header", action="store_true", help="Show small header on right panel (e.g., frame index).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir).resolve()
    anno_dir = Path(args.anno_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir not found: {video_dir}")
    if not anno_dir.exists():
        raise FileNotFoundError(f"anno_dir not found: {anno_dir}")

    keep_audio = args.keep_audio and (not args.no_keep_audio)
    exts = (args.video_ext.lower(),)

    videos = iter_videos(video_dir, exts)
    if not videos:
        print(f"[INFO] No videos found under: {video_dir} (ext={exts})")
        return

    iterator = tqdm(videos, desc="Processing") if (HAS_TQDM and len(videos) > 1) else videos

    ok_cnt = 0
    skip_cnt = 0
    fail_cnt = 0

    for vpath in iterator:
        rel = vpath.relative_to(video_dir)
        jpath = (anno_dir / rel).with_suffix(args.anno_ext)
        outpath = (out_dir / rel).with_suffix(args.video_ext)

        if not jpath.exists():
            skip_cnt += 1
            print(f"[WARN] Missing annotation, skip: {jpath}")
            continue

        success = process_one_video(
            video_path=vpath,
            json_path=jpath,
            out_path=outpath,
            overwrite=args.overwrite,
            keep_audio=keep_audio,
            font_scale=args.font_scale,
            thickness=args.thickness,
            margin=args.margin,
            line_spacing=args.line_spacing,
            fourcc=args.fourcc,
            show_header=args.show_header,
        )
        if success:
            ok_cnt += 1
        else:
            fail_cnt += 1

    print(f"[DONE] ok={ok_cnt}, skipped(no json)={skip_cnt}, failed={fail_cnt}")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()