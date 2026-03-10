#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.pytorch_i3d import InceptionI3d


def initialize_i3d(weights_path: str, device: torch.device):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    state = torch.load(weights_path, map_location="cpu")
    i3d.load_state_dict(state)
    i3d.to(device)
    i3d.eval()
    return i3d


def _center_crop_resize_rgb(frame_bgr, target_size=256, crop_size=224):
    frame = cv2.resize(frame_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    start = (target_size - crop_size) // 2
    frame = frame[start:start + crop_size, start:start + crop_size]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = frame.transpose(2, 0, 1)  # (C,H,W)
    return frame


def read_video_frames_30fps(video_path: str, target_fps=30.0, target_size=256, crop_size=224, max_keep_frames=None):
    """
    Read video frames and resample to target_fps by timestamps (best for downsampling like 60->30).
    Returns: np.ndarray (T, C, H, W) float32 in [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps is None or orig_fps <= 1e-3:
        orig_fps = None  # fallback to POS_MSEC or frame index logic

    frames = []
    keep_dt = 1.0 / float(target_fps)
    next_t = 0.0

    frame_idx = 0
    last_pos_msec = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Current timestamp (seconds)
        if orig_fps is not None:
            t = frame_idx / float(orig_fps)
        else:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec is not None and pos_msec > 0:
                t = pos_msec / 1000.0
                last_pos_msec = pos_msec
            else:
                # worst-case fallback
                t = frame_idx * keep_dt

        if t + 1e-9 >= next_t:
            frames.append(_center_crop_resize_rgb(frame, target_size, crop_size))
            next_t += keep_dt
            if max_keep_frames is not None and len(frames) >= max_keep_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32)  # (T,C,H,W)
    return arr


def prepare_input_tensor(frames_TCHW: np.ndarray) -> torch.Tensor:
    # (T,C,H,W) -> (1,C,T,H,W)
    t, c, h, w = frames_TCHW.shape
    arr = frames_TCHW.transpose(1, 0, 2, 3).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor


@torch.no_grad()
def extract_features(i3d, frames_TCHW: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Returns features of shape (seq_len, feat_dim) float32
    """
    input_tensor = prepare_input_tensor(frames_TCHW).to(device, non_blocking=True)
    feats = i3d.extract_features(input_tensor)  # usually (1, 1024, T', 1, 1)
    seq_len = feats.shape[2]
    feats = feats.squeeze(0).permute(1, 2, 3, 0).reshape(seq_len, -1)  # (T', D)
    return feats.detach().cpu().numpy().astype(np.float32)


def video_cache_path(cache_dir: Path, video_path: Path) -> Path:
    # As requested: basename only
    return cache_dir / f"{video_path.stem}.npy"


def get_or_compute_feats(
    video_path: Path,
    cache_dir: Path,
    i3d,
    device: torch.device,
    target_fps: float,
    target_size: int,
    crop_size: int,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_path = video_cache_path(cache_dir, video_path)

    if save_path.exists():
        feats = np.load(save_path)
        return feats, save_path, True

    frames = read_video_frames_30fps(
        str(video_path),
        target_fps=target_fps,
        target_size=target_size,
        crop_size=crop_size,
    )
    feats = extract_features(i3d, frames, device=device)
    np.save(save_path, feats)
    return feats, save_path, False


def l2_normalize(x: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def video_embedding_from_feats(feats: np.ndarray) -> np.ndarray:
    # feats: (T', D) -> (D,)
    emb = feats.mean(axis=0)
    emb = l2_normalize(emb)
    return emb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/network_space/server126/shared/sunyx/datasets/custom/data/bili_solo_candidates/videos",
        help="库视频文件夹（递归找 .mp4）",
    )
    parser.add_argument("--query_video", type=str, required=True, help="要匹配的 .mp4 路径")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/network_space/server126/shared/sunyx/datasets/custom/data/bili_solo_candidates/i3d_cache",
        help="特征缓存文件夹（basename.npy）",
    )
    parser.add_argument("--weights_path", type=str, default="utils/rgb_charades.pt", help="I3D 权重路径")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip_if_same_path", action="store_true", help="若 query_video 在库中，是否跳过它")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    query_video = Path(args.query_video)
    cache_dir = Path(args.cache_dir)

    if not query_video.exists():
        raise FileNotFoundError(f"query_video not found: {query_video}")
    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir not found: {video_dir}")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Info] device = {device}")

    # collect library videos
    lib_videos = sorted(video_dir.rglob("*.mp4"))
    if len(lib_videos) == 0:
        raise RuntimeError(f"No .mp4 found under: {video_dir}")
    print(f"[Info] found {len(lib_videos)} library videos under {video_dir}")

    # collision warning (basename-only cache)
    stems = {}
    collisions = 0
    for vp in lib_videos:
        if vp.stem in stems and stems[vp.stem] != vp:
            collisions += 1
        else:
            stems[vp.stem] = vp
    if collisions > 0:
        print(f"[Warn] Detected {collisions} potential basename collisions in library. "
              f"Cache may be overwritten because cache uses basename.npy only.")

    i3d = initialize_i3d(args.weights_path, device)

    # query feats + emb
    q_feats, q_cache_path, q_loaded = get_or_compute_feats(
        query_video, cache_dir, i3d, device,
        target_fps=args.target_fps,
        target_size=args.target_size,
        crop_size=args.crop_size,
    )
    q_emb = video_embedding_from_feats(q_feats)
    print(f"[Info] query feats: {q_feats.shape}, cache: {q_cache_path}, loaded={q_loaded}")

    # iterate library
    sims = []
    for vp in tqdm(lib_videos, desc="Compute/Load library feats"):
        try:
            if args.skip_if_same_path and vp.resolve() == query_video.resolve():
                continue

            feats, save_path, loaded = get_or_compute_feats(
                vp, cache_dir, i3d, device,
                target_fps=args.target_fps,
                target_size=args.target_size,
                crop_size=args.crop_size,
            )
            emb = video_embedding_from_feats(feats)
            sim = float(np.dot(q_emb, emb))  # cosine because both normalized
            sims.append((sim, str(vp), str(save_path), loaded, feats.shape[0]))

        except Exception as e:
            print(f"[Error] {vp}: {e}")

    if len(sims) == 0:
        raise RuntimeError("No valid videos processed in library (all failed?).")

    sims.sort(key=lambda x: x[0], reverse=True)
    topk = sims[: args.topk]

    print("\n===== Top Similar Videos =====")
    for rank, (sim, vp, cachep, loaded, tlen) in enumerate(topk, 1):
        print(f"{rank:02d}. sim={sim:.4f}  T'={tlen:<4d}  loaded={loaded}  video={vp}")
    print("==============================\n")


if __name__ == "__main__":
    main()