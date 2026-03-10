import os
import json
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def detect_downbeats_madmom(audio_path):
    """
    使用 madmom 检测音频中的 downbeats
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        downbeat_times: downbeat 时间点数组（秒）
        total_duration: 音频总时长（秒）
    """
    total_duration = librosa.get_duration(path=audio_path)
    act = RNNDownBeatProcessor()(audio_path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=100)
    db_out = proc(act)
    downbeat_times = db_out[db_out[:, 1] == 1][:, 0]
    if len(downbeat_times) > 0:
        downbeat_times[0] = 0.0
    return downbeat_times, total_duration


def segment_audio_by_downbeats(audio_path, downbeats_per_segment=4):
    """
    根据 downbeats 对音频进行分段
    
    Args:
        audio_path: 音频文件路径
        downbeats_per_segment: 每个 segment 包含的 downbeat 数量
        
    Returns:
        dict: 包含 downbeats_per_segment 和 segments 的字典
    """
    # 检测 downbeats
    downbeat_times, total_duration = detect_downbeats_madmom(audio_path)
    
    # 每 downbeats_per_segment 个 downbeat 作为一个分段点
    segment_times = downbeat_times[::downbeats_per_segment].tolist()
    segment_times.append(total_duration)
    
    # 构建 segments 列表
    segments = []
    for i in range(len(segment_times) - 1):
        segments.append({
            "start_sec": float(segment_times[i]),
            "end_sec": float(segment_times[i + 1])
        })
    
    return {
        "downbeats_per_segment": downbeats_per_segment,
        "segments": segments
    }


def process_single_audio(audio_path, output_dir, downbeats_per_segment=4):
    """
    处理单个音频文件
    
    Args:
        audio_path: 音频文件路径
        output_dir: 输出目录
        downbeats_per_segment: 每个 segment 包含的 downbeat 数量
    """
    try:
        basename = Path(audio_path).stem
        output_path = os.path.join(output_dir, f"{basename}.json")
        
        # 如果输出文件已存在，跳过
        if os.path.exists(output_path):
            print(f"Skip (already exists): {basename}")
            return
        
        # 进行分段
        result = segment_audio_by_downbeats(audio_path, downbeats_per_segment)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Success: {basename} -> {len(result['segments'])} segments")
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        import traceback
        traceback.print_exc()


def process_directory(input_dir, output_dir, downbeats_per_segment=4):
    """
    处理目录中的所有 .mp3 文件
    
    Args:
        input_dir: 输入目录，包含 .mp3 文件
        output_dir: 输出目录
        downbeats_per_segment: 每个 segment 包含的 downbeat 数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 .mp3 文件
    audio_files = list(Path(input_dir).glob("*.mp3"))
    
    if not audio_files:
        print(f"No .mp3 files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} .mp3 files")
    print(f"Output directory: {output_dir}")
    print(f"Downbeats per segment: {downbeats_per_segment}")
    print("-" * 60)
    
    # 处理每个音频文件
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        process_single_audio(str(audio_path), output_dir, downbeats_per_segment)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment audio files by downbeats")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录，包含 .mp3 文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，保存 .json 分段结果"
    )
    parser.add_argument(
        "--downbeats_per_segment",
        type=int,
        default=4,
        help="每个 segment 包含的 downbeat 数量（默认：4）"
    )
    
    args = parser.parse_args()
    
    process_directory(
        args.input_dir,
        args.output_dir,
        args.downbeats_per_segment
    )
