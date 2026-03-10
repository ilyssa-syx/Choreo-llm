import os
import json
import re
import librosa
import numpy as np
import traceback
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm  # 建议安装 tqdm: pip install tqdm，用于显示进度条

# 假设这些库你已经正确安装和配置
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from utils.genre_extraction import extract_genre_from_filename
from utils.chord_progressions import extract_chord_progression
from utils.key_extraction import extract_key
from utils.bpm_extraction import get_tempo
from utils.call_api import call_gemini_api
# from utils.call_musicflamingo import call_musicflamingo

# ================= 原有辅助函数保持不变 =================

def detect_downbeats_madmom(audio_path):
    # ... (保持原样，为了节省篇幅省略，请保留你原来的代码) ...
    total_duration = librosa.get_duration(path=audio_path)
    act = RNNDownBeatProcessor()(audio_path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=100)
    db_out = proc(act)
    downbeat_times = db_out[db_out[:,1] == 1][:,0]
    if len(downbeat_times) > 0:
        downbeat_times[0] = 0.0
    return downbeat_times, total_duration

def return_periods(merged_path, wav_path, downbeats_per_segment=4):
    # ... (保持原样) ...
    # 注意：print 语句在多进程中可能会乱序，建议减少非必要的 print
    with open(merged_path, 'r') as f:
        captions = json.load(f)

    downbeat_times, trim_secs = detect_downbeats_madmom(wav_path)
    
    segment_times = downbeat_times[::downbeats_per_segment].tolist()
    segment_times = np.array(segment_times + [trim_secs])
    
    segment_cnt = 0
    song_cap = []
    sentence_cap = []
    for caption in captions:
        if caption['end_frame'] / 30.0 <= segment_times[segment_cnt + 1]:
            sentence_cap.append(caption)
        elif caption['start_frame'] / 30.0 >= segment_times[segment_cnt + 1]:
            song_cap.append(sentence_cap.copy())
            segment_cnt += 1
            sentence_cap = [caption]
            
    if len(sentence_cap) > 0:
        song_cap.append(sentence_cap.copy())

    return segment_times, song_cap

def generate_segment_summary(segment):
    # ... (保持原样) ...
    tags = []
    for atom in segment:
        modifier = atom.get('modifier', {})
        simple_tag = modifier.get('simple_tag', '')
        tags.append(simple_tag)

    all_tags = ""
    for idx, tag in enumerate(tags, start=1):
        all_tags += f"Movement {idx}: {tag}. \n"

    system_message = (
        "You are an expert dance analyst and choreographer focusing on the 'big picture' of movement. "
        "Summarize dance sequences by capturing the overall kinetic arc and mood, rather than listing mechanical steps. "
        "Since the input data may be sparse, prioritize continuity: bridge gaps by focusing on the flow of energy and body weight. "
        "Abstract specific limb details into broader movement themes (e.g., 'sweeping extensions' instead of 'arm raises'). "
        "Use torso rotation as a key structural guide: if the body orientation alternates, describe this as a rhythmic or thematic alternation. "
        "Write in a fluid, evocative, and musical voice that conveys the expressive spirit (e.g., lyrical, percussive, expansive). "
        "Keep your answer within 75 words, in a single paragraph without bullet points."
    )
    prompt = (
        "Write a flowing, atmospheric summary (max 75 words) that captures the essence of the sequence. "
        "Do not simply list actions; instead, weave the sparse descriptions into a cohesive narrative about energy and direction. "
        "Prioritize general movement qualities (e.g., 'dynamic shifts', 'fluid rotations', 'sharp strikes') over precise anatomical details. "
        "Use the torso's rotation pattern as the backbone of the structure: if the orientation shifts left/right, frame the whole phrase as a rhythmic exchange or alternating motif, using phrases like 'echoes on the other side' or 'sways between sides'. "
        "Allow the description to be impressionistic—focus on how the movement *feels* and travels through space. "
        "If details are missing, focus on the mood (e.g., 'driving', 'suspended', 'playful'). "
        "Avoid step-by-step accounting.\n\n"
        "Movement cues (for reference):\n"
        f"{all_tags}"
    )

    summary = call_gemini_api(system=system_message, prompt=prompt, model='gemini-2.5-pro')
    return summary

def summarize_segment(segment, segment_start, segment_end, wav_path):
    # ... (保持原样) ...
    m_index = os.path.basename(wav_path).find('m')
    genre = extract_genre_from_filename(os.path.basename(wav_path)[m_index:m_index+3])
    key = extract_key(wav_path, segment_start, segment_end)
    chord_progression = extract_chord_progression(wav_path, segment_start, segment_end)
    bpm = get_tempo(os.path.basename(wav_path)[m_index:m_index+4])

    summary = (
        f"This is a {genre}-style music segment lasting {(segment_end - segment_start):.2f} seconds, "
        f"in key {key}, with chord progression {chord_progression}, and tempo {bpm} BPM. "
    )
    summary += (
        "As a choreographer preparing movement for this music, I will consider the musical style, "
        "rhythmic feel, and mood (as implied by the key), as well as the musical development indicated by "
        "the chord progression, when arranging motion and transitions."
    )
    summary += "\n\nHigh-level movement synthesis:\n"
    summary += generate_segment_summary(segment)

    summary += "\n\nSpecific movement instructions for the model:\n\n"
    for idx, atom in enumerate(segment, start=1):
        modifier = atom.get('modifier', {})
        start_sec = atom['start_frame'] / 30
        end_sec = atom['end_frame'] / 30
        summary += f"***Movement {idx}***: {(start_sec-segment_start):.2f} - {(end_sec-segment_start):.2f}\n"
        summary += f"simple tag: {modifier.get('simple_tag', '')}\n"
        summary += f"whole body: {modifier.get('whole_body', '')}\n"
        summary += f"lower half body: {modifier.get('lower_half_body', '')}\n"
        summary += f"upper half body: {modifier.get('upper_half_body', '')}\n"
        summary += f"torso: {modifier.get('torso', '')}\n\n"

    return summary

# ================= 新增/修改后的并行处理逻辑 =================

def process_single_file(file_name, merged_folder, wav_folder, out_folder, downbeats_per_segment):
    """
    处理单个文件的函数，将被多进程调用。
    """
    try:
        out_path = os.path.join(out_folder, file_name)
        
        # 1. 检查是否已存在（跳过逻辑）
        if os.path.exists(out_path):
            return f"Skipped (exists): {file_name}"

        if not file_name.endswith('.json'):
            return None

        merged_path = os.path.join(merged_folder, file_name)
        wav_name = os.path.splitext(file_name)[0]
        wav_path = os.path.join(wav_folder, wav_name+'.wav')

        # 检查 wav 是否存在
        if not os.path.exists(wav_path):
            return f"Error: Wav file not found for {file_name}"

        # 2. 核心处理逻辑
        segment_times, segments = return_periods(merged_path, wav_path, downbeats_per_segment)
        
        augmented_segments = []
        for i in range(len(segments)):
            segment = segments[i]
            augmented_segment = {}
            augmented_segment['sentence'] = segment.copy()
            
            # API 调用在这里发生
            segment_summary = summarize_segment(
                segment=segment, 
                segment_start=segment_times[i], 
                segment_end=segment_times[i+1], 
                wav_path=wav_path
            )
            
            augmented_segment['sentence_summary'] = segment_summary
            augmented_segment['start_sec'] = segment_times[i]
            augmented_segment['end_sec'] = segment_times[i+1]
            
            augmented_segments.append(augmented_segment)
        
        # 3. 保存结果
        with open(out_path, 'w') as f:
            json.dump(augmented_segments, f, indent=4)
        
        return f"Processed: {file_name} ({len(segments)} segments)"

    except Exception as e:
        # 捕获异常防止单个文件错误导致整个程序崩溃
        error_msg = traceback.format_exc()
        return f"Failed {file_name}: {e}\n{error_msg}"

def deal_folder_parallel(merged_folder, wav_folder, out_folder, downbeats_per_segment=4, max_workers=8):
    """
    并行处理文件夹。
    max_workers: 并行进程数，建议设置为 CPU 核心数，或者根据 API 限流情况调整。
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # 获取所有需要处理的文件列表
    all_files = [f for f in os.listdir(merged_folder) if f.endswith('.json')]
    
    print(f"Starting parallel processing of {len(all_files)} files with {max_workers} workers...")
    

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 partial 固定住那些不变的参数
        worker_func = partial(
            process_single_file, 
            merged_folder=merged_folder, 
            wav_folder=wav_folder, 
            out_folder=out_folder, 
            downbeats_per_segment=downbeats_per_segment
        )
        
        # 提交任务
        futures = [executor.submit(worker_func, file_name) for file_name in all_files]
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Processing"):
            result = future.result()
            if result:
                # 如果你想看每个文件的详细结果，可以取消注释下面这行，但在进度条下可能显示混乱
                # tqdm.write(result)
                if "Failed" in result or "Error" in result:
                    tqdm.write(result) # 只打印错误信息

if __name__ == "__main__":
    merged_folder = '../../aist_annotation/gemini_caption/merged/test'
    wav_folder = '/network_space/storage43/sunyixuan/models/EDGE/data/test/wavs'
    out_folder = './split/test'
    
    # 建议 max_workers 设置为 4 到 8 之间。
    # 如果设置过大，可能会触发 Gemini API 的 Rate Limit (429 Error)。
    deal_folder_parallel(merged_folder, wav_folder, out_folder, downbeats_per_segment=4, max_workers=32)