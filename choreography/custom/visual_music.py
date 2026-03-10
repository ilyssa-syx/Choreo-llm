#!/usr/bin/env python3
"""
Find top-k videos whose audio is nearest to a query audio using CLAP embeddings.

Changes vs "first-10s only":
- Uniformly sample `num_segments` clips across the full audio (optionally skipping head),
  compute CLAP embedding per clip, then mean-pool + L2 normalize for final vector.
- Cache stores the *sequence* of segment embeddings: shape (num_segments, emb_dim).

Dependencies:
  pip install laion-clap
  ffmpeg + ffprobe available in PATH
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm
import shutil

DEFAULT_EXTS = ".mp4"
DEFAULT_TOP_K = 200
DEFAULT_TARGET_SR = 48000
DEFAULT_CLIP_SEC = 10.0
DEFAULT_NUM_SEGMENTS = 5
DEFAULT_SKIP_HEAD_SEC = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top-k nearest videos by CLAP audio embedding (multi-segment uniform sampling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query_mp3", default=None, help="Path to query audio file (.mp3/.wav/...)")
    parser.add_argument("--video_dir", required=True, help="Directory containing video files")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--target_sr", type=int, default=DEFAULT_TARGET_SR)
    parser.add_argument("--clip_sec", type=float, default=DEFAULT_CLIP_SEC, help="Seconds per sampled clip")
    parser.add_argument(
        "--num_segments",
        type=int,
        default=DEFAULT_NUM_SEGMENTS,
        help="Uniformly sample this many clips across the audio (suggest 3~5)",
    )
    parser.add_argument(
        "--skip_head_sec",
        type=float,
        default=DEFAULT_SKIP_HEAD_SEC,
        help="Lower bound of sampling window (e.g., skip silent intro). Sampling starts from this time.",
    )

    parser.add_argument("--exts", default=DEFAULT_EXTS, help="Comma-separated list of video extensions to scan")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan video_dir")

    parser.add_argument("--cache_dir", default=None, help="Optional cache directory for embeddings (.npy)")
    parser.add_argument("--build_cache", action="store_true", help="Only build cache for videos in video_dir and exit")
    parser.add_argument("--cache_only", action="store_true", help="Only use cached features for videos; skip cache misses")

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--amodel", default="HTSAT-tiny", help="CLAP audio encoder arch")
    parser.add_argument("--fusion", action="store_true", help="Use fusion CLAP model (larger/slower)")
    parser.add_argument("--model_id", type=int, default=-1, help="Checkpoint id for laion_clap.load_ckpt")
    parser.add_argument("--clap_ckpt", default=None, help="Local path to laion_clap .pt checkpoint (avoid online download)")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Where to copy top results. Default: ./music_knn/<query_basename>/",
    )
    parser.add_argument(
        "--copy_top_k",
        type=int,
        default=5,
        help="Copy top-N nearest videos to out_dir",
    )
    return parser.parse_args()


def iter_video_files(video_dir: Path, exts: set[str], recursive: bool) -> Iterable[Path]:
    if recursive:
        for path in sorted(video_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                yield path
    else:
        for path in sorted(video_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def run_cmd(cmd: list[str]) -> Tuple[int, bytes, bytes]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


def get_media_duration_sec(path: Path) -> float:
    """
    Use ffprobe to get duration (seconds). Returns 0.0 on failure.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        return 0.0
    try:
        dur = float(out.decode("utf-8", "ignore").strip())
        return max(0.0, dur)
    except Exception:
        return 0.0


def decode_audio_ffmpeg(path: Path, target_sr: int, start_sec: float, clip_sec: float) -> np.ndarray:
    """
    Decode a mono float32 waveform via ffmpeg, resampled to target_sr,
    for [start_sec, start_sec+clip_sec).
    """
    if clip_sec <= 0:
        raise ValueError(f"clip_sec must be > 0, got {clip_sec}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, start_sec)),
        "-t",
        str(clip_sec),
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffmpeg decode failed ({rc}): {path}\n{err.decode('utf-8', 'ignore')}")
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"Empty audio after decoding: {path} @ start={start_sec:.2f}s")
    return audio


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / (n + eps)


def uniform_segment_starts(duration: float, clip_sec: float, num_segments: int, skip_head_sec: float) -> list[float]:
    """
    Uniformly sample start times within [skip_head_sec, duration-clip_sec].
    If duration is too short, return [0.0].
    """
    if duration <= 0 or duration <= clip_sec or num_segments <= 1:
        return [max(0.0, min(skip_head_sec, max(0.0, duration - clip_sec))) if duration > 0 else 0.0]

    max_start = max(0.0, duration - clip_sec)
    lo = max(0.0, min(skip_head_sec, max_start))

    if max_start <= lo + 1e-6:
        return [lo]

    # linspace includes both ends -> nicely covers the whole audio span
    starts = np.linspace(lo, max_start, num_segments, dtype=np.float32).tolist()
    return [float(s) for s in starts]


def cache_key(
    path: Path,
    target_sr: int,
    clip_sec: float,
    num_segments: int,
    skip_head_sec: float,
    amodel: str,
    fusion: bool,
    model_id: int,
) -> str:
    st = path.stat()
    raw = (
        f"{path.resolve()}|size={st.st_size}|mtime_ns={st.st_mtime_ns}"
        f"|sr={target_sr}|clip={clip_sec}|K={num_segments}|skip={skip_head_sec}"
        f"|amodel={amodel}|fusion={fusion}|model_id={model_id}|pool=mean"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_segment_embeddings(
    path: Path,
    clap_model,
    target_sr: int,
    clip_sec: float,
    num_segments: int,
    skip_head_sec: float,
) -> np.ndarray:
    """
    Returns segment embedding sequence: (S, D) float32, each row L2-normalized.
    """
    duration = get_media_duration_sec(path)
    starts = uniform_segment_starts(duration, clip_sec, num_segments, skip_head_sec)

    seg_embs: list[np.ndarray] = []
    for s in starts:
        audio = decode_audio_ffmpeg(path, target_sr=target_sr, start_sec=s, clip_sec=clip_sec)
        audio_batch = np.expand_dims(audio, 0)  # (1, T)
        emb = clap_model.get_audio_embedding_from_data(audio_batch, use_tensor=False)  # (1, D)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        emb = l2_normalize(emb)
        seg_embs.append(emb)

    seq = np.stack(seg_embs, axis=0).astype(np.float32)  # (S, D)
    return seq


def pool_mean(seq: np.ndarray) -> np.ndarray:
    """
    seq: (S, D) with per-segment L2-normalized embeddings
    return: (D,) mean pooled then L2-normalized
    """
    v = np.mean(seq, axis=0).astype(np.float32)
    return l2_normalize(v)


def load_or_extract_embedding_seq(
    path: Path,
    clap_model,
    target_sr: int,
    clip_sec: float,
    num_segments: int,
    skip_head_sec: float,
    cache_dir: Optional[Path],
    cache_only: bool,
    amodel: str,
    fusion: bool,
    model_id: int,
) -> np.ndarray:
    """
    Returns cached/extracted sequence embeddings: (S, D).
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key(path, target_sr, clip_sec, num_segments, skip_head_sec, amodel, fusion, model_id)}.npy"
        if cache_path.exists():
            seq = np.load(cache_path)
            if seq.ndim == 2:
                return seq.astype(np.float32)
            if cache_only:
                raise FileNotFoundError(f"Cache shape mismatch: {cache_path} (shape={seq.shape})")
        if cache_only:
            raise FileNotFoundError(f"Cache miss: {cache_path}")

    seq = extract_segment_embeddings(
        path,
        clap_model=clap_model,
        target_sr=target_sr,
        clip_sec=clip_sec,
        num_segments=num_segments,
        skip_head_sec=skip_head_sec,
    )

    if cache_path is not None:
        np.save(cache_path, seq)

    return seq


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> None:
    args = parse_args()

    if args.num_segments < 1:
        raise ValueError("--num_segments must be >= 1")
    if args.clip_sec <= 0:
        raise ValueError("--clip_sec must be > 0")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import laion_clap  # type: ignore

    device_str = "cuda:0" if (args.device == "cuda") else "cpu"
    clap_model = laion_clap.CLAP_Module(
        enable_fusion=bool(args.fusion),
        device=device_str,
        amodel=args.amodel,
    )
    if args.clap_ckpt:
        clap_model.load_ckpt(ckpt=args.clap_ckpt)
    else:
        clap_model.load_ckpt(model_id=args.model_id)
    
    video_dir = Path(args.video_dir)
    exts = {ext.strip().lower() for ext in args.exts.split(",") if ext.strip()}
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if not video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")

    candidates = list(iter_video_files(video_dir, exts, args.recursive))

    if args.build_cache:
        if cache_dir is None:
            raise ValueError("--build_cache requires --cache_dir")
        for path in tqdm(candidates, desc="Building cache (segment seq)"):
            try:
                _ = load_or_extract_embedding_seq(
                    path,
                    clap_model=clap_model,
                    target_sr=args.target_sr,
                    clip_sec=args.clip_sec,
                    num_segments=args.num_segments,
                    skip_head_sec=args.skip_head_sec,
                    cache_dir=cache_dir,
                    cache_only=False,
                    amodel=args.amodel,
                    fusion=bool(args.fusion),
                    model_id=args.model_id,
                )
            except Exception as exc:
                print(f"[WARN] Skip {path}: {exc}")
        print("Cache build complete.")
        return

    if args.query_mp3 is None:
        raise ValueError("Need --query_mp3 unless --build_cache is set.")

    query_path = Path(args.query_mp3)
    if not query_path.exists():
        raise FileNotFoundError(f"Query audio not found: {query_path}")

    # query: seq -> pooled
    query_seq = load_or_extract_embedding_seq(
        query_path,
        clap_model=clap_model,
        target_sr=args.target_sr,
        clip_sec=args.clip_sec,
        num_segments=args.num_segments,
        skip_head_sec=args.skip_head_sec,
        cache_dir=cache_dir,
        cache_only=False,
        amodel=args.amodel,
        fusion=bool(args.fusion),
        model_id=args.model_id,
    )
    query_vec = pool_mean(query_seq)

    import heapq
    heap: list[tuple[float, Path]] = []

    for path in tqdm(candidates, desc="Encoding videos"):
        try:
            seq = load_or_extract_embedding_seq(
                path,
                clap_model=clap_model,
                target_sr=args.target_sr,
                clip_sec=args.clip_sec,
                num_segments=args.num_segments,
                skip_head_sec=args.skip_head_sec,
                cache_dir=cache_dir,
                cache_only=args.cache_only,
                amodel=args.amodel,
                fusion=bool(args.fusion),
                model_id=args.model_id,
            )
            vec = pool_mean(seq)
        except Exception as exc:
            print(f"[WARN] Skip {path}: {exc}")
            continue

        sim = cosine(query_vec, vec)

        if len(heap) < args.top_k:
            heapq.heappush(heap, (sim, path))
        else:
            if sim > heap[0][0]:
                heapq.heapreplace(heap, (sim, path))

    top = sorted(heap, key=lambda x: x[0], reverse=True)

    print("\nTop results:")
    if not top:
        print("No valid candidates found. Check cache/audio/ffmpeg.")
        return

        # ===== Copy top-N to output folder =====
    query_stem = query_path.stem  # e.g., 132 from 132.mp3
    out_dir = Path(args.out_dir) if args.out_dir else (Path("./music_knn") / query_stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_copy = max(0, min(args.copy_top_k, len(top)))
    for rank, (sim, src_path) in enumerate(top[:n_copy], start=1):
        dst_path = out_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as exc:
            print(f"[WARN] Failed to copy {src_path} -> {dst_path}: {exc}")

    print(f"\nCopied top {n_copy} videos to: {out_dir.resolve()}")
    


if __name__ == "__main__":
    main()