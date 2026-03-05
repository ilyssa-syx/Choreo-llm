#!/usr/bin/env python3
# coding: utf-8

import os
import json
import glob
import gc
import subprocess
import tempfile
import logging
import torch
import cv2
import numpy as np
from utils.i3d_feature_extractor import initialize_i3d, extract_features
from utils.similarity_matrix import process_features
from utils.get_segmentation import get_block_starts

# -----------------------
# 用户可修改的路径/配置
# -----------------------
input_dir = ""
OUTPUT_DIR = ""

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][PID %(process)d][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# -----------------------
# 辅助函数
# -----------------------

def safe_mkdir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def convert_to_30fps(input_path: str) -> str:
    """
    用 ffmpeg 将视频转换为 30fps，返回临时文件路径。
    调用方负责删除该临时文件。
    """
    suffix = os.path.splitext(input_path)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', 'fps=30', '-r', '30',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-an',
        tmp.name
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name


def read_and_slice_video(video_path, stride, length, num_slices, target_size=256, crop_size=224):
    """
    读取视频并按照指定参数切片。视频必须是 30fps。

    Args:
        video_path: 视频文件路径（30fps）
        stride: 切片步长(秒)
        length: 每个切片长度(秒)
        num_slices: 最大切片数量
        target_size: resize的目标尺寸
        crop_size: 中心裁剪的尺寸

    Returns:
        list: 包含多个视频片段的列表，每个片段是 numpy array (T, C, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - 30.0) < 0.5, \
        f"视频帧率应为 30fps，但实际为 {fps:.2f}fps: {video_path}"

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame 的典型形状是 (H, W, 3)，dtype 通常是 uint8，颜色顺序是 BGR（OpenCV 默认）。
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        start = (target_size - crop_size) // 2
        frame = frame[start:start + crop_size, start:start + crop_size]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 临时测试：没有归一化的差异
        # frame = (frame.astype(np.float32) / 255.0)
        frame = (frame.astype(np.float32) / 255.0) * 2 - 1  # 归一化到 [-1, 1]
        frame = frame.transpose(2, 0, 1)  # (C, H, W)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")

    all_frames = np.stack(frames, axis=0)

    logging.info(f'Video path: {video_path}')
    logging.info(f'Total frames (30fps): {all_frames.shape[0]}')

    video_slices = []
    window = int(length * 30)
    stride_step = int(stride * 30)
    start_idx = 0
    slice_count = 0

    while start_idx <= len(all_frames) - window and slice_count < num_slices:
        video_slice = all_frames[start_idx: start_idx + window]
        video_slices.append(video_slice)
        logging.info(f'Slice {slice_count}: frames {start_idx} to {start_idx + window}, shape: {video_slice.shape}')
        start_idx += stride_step
        slice_count += 1

    logging.info(f'Total slices created: {len(video_slices)}')
    return video_slices


# -----------------------
# 处理单个文件
# -----------------------

def process_single_file(file_rel_no_ext: str, i3d):
    """处理单个视频文件，先转 30fps 再切片提特征"""
    prefix = f"[{file_rel_no_ext}]"
    input_path = os.path.join(input_dir, file_rel_no_ext + ".mp4")
    logging.info(f"{prefix} 开始处理视频: {input_path}")

    tmp_path = None
    try:
        logging.info(f"{prefix} 转换为 30fps ...")
        tmp_path = convert_to_30fps(input_path)

        video_slices = read_and_slice_video(
            tmp_path,
            stride=0.5,
            length=5,
            num_slices=1000
        )

        safe_mkdir(OUTPUT_DIR)
        slices_counted = 0

        for i, chunk in enumerate(video_slices):
            try:
                output_path = os.path.join(OUTPUT_DIR, file_rel_no_ext + f"_slice{i}.json")
                safe_mkdir(os.path.dirname(output_path))

                if os.path.exists(output_path):
                    logging.info(f"{prefix} slice {i} 已存在，跳过 -> {output_path}")
                    slices_counted += 1
                    continue

                feat = extract_features(i3d, chunk)
                labels_s = process_features(feat)
                block_starts = get_block_starts(labels_s)

                output = []
                motion_num = len(block_starts)
                block_starts.append(150)

                for j in range(motion_num):
                    output.append({
                        "motion": j,
                        "start_frame": int(block_starts[j]),
                        "end_frame": int(block_starts[j + 1])
                    })

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=4, ensure_ascii=False)

                slices_counted += 1
                logging.info(f"{prefix} slice {i} saved -> {output_path}")

            except Exception as e:
                logging.exception(f"{prefix} 处理 slice {i} 失败: {e}")

        logging.info(f"{prefix} ✓ Completed {slices_counted} slices")
        return slices_counted

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# -----------------------
# 主入口
# -----------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Single-GPU video segmentation")
    parser.add_argument('--input_dir', type=str, default=input_dir,
                        help='输入视频目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出 JSON 目录')
    parser.add_argument('--filter_list', type=str, default=None,
                        help='可选：只处理 basename 在此 txt 文件中的视频（每行一个 basename，不含扩展名）')
    args = parser.parse_args()

    input_dir = args.input_dir
    OUTPUT_DIR = args.output_dir

    logging.info("=" * 60)
    logging.info("Single-GPU Video Processing")
    logging.info("=" * 60)

    all_videos = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
    files = sorted([os.path.splitext(os.path.relpath(v, input_dir))[0] for v in all_videos])

    logging.info(f"Found {len(files)} video files in {input_dir}")

    # 按 filter_list 过滤：只保留 basename 在白名单中的文件
    if args.filter_list:
        with open(args.filter_list, 'r', encoding='utf-8') as fp:
            allowed = [line.strip() for line in fp if line.strip()]
        before = len(files)
        allowed_set = set(allowed)
        files = [f for f in files if os.path.basename(f).replace('_c01_', '_cAll_') in allowed_set]
        logging.info(f"filter_list: {args.filter_list} ({len(allowed)} entries)")
        logging.info(f"Filtered {before} -> {len(files)} files")

    if not files:
        logging.warning("没有找到需要处理的视频文件")
        raise SystemExit(0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    i3d = initialize_i3d()

    total_slices = 0
    failed_files = []

    for idx, file_rel_no_ext in enumerate(files):
        logging.info(f"\nProgress: [{idx + 1}/{len(files)}]")
        try:
            slices = process_single_file(file_rel_no_ext, i3d)
            total_slices += slices
        except Exception as e:
            logging.exception(f"处理失败: {file_rel_no_ext}: {e}")
            failed_files.append(file_rel_no_ext)

    # 汇总
    logging.info("\n" + "=" * 60)
    logging.info("Processing Summary")
    logging.info("=" * 60)
    logging.info(f"Total files processed: {len(files)}")
    logging.info(f"Successful: {len(files) - len(failed_files)}")
    logging.info(f"Failed: {len(failed_files)}")
    logging.info(f"Total slices generated: {total_slices}")
    logging.info("=" * 60)

    if failed_files:
        logging.info("Failed Files:")
        for f in failed_files:
            logging.info(f"  {f}")
        with open('failed_files.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(failed_files))
        logging.info("Failed files saved to: failed_files.txt")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("\n✓ All processing completed!")
    logging.info(f"Results saved to: {OUTPUT_DIR}/")
