import json
import os
import re

from utils.genre_extraction import extract_genre_from_filename
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from utils.call_musicflamingo import call_musicflamingo


# ===== 新增：模型管理函数 =====

def download_and_save_model(model_id="nvidia/music-flamingo-hf", local_dir="./models/music_flamingo", clean_tmp=True):
    """
    首次运行：下载模型并保存到本地目录
    
    Args:
        model_id: HuggingFace 模型ID
        local_dir: 本地保存目录
        clean_tmp: 是否在下载完成后清理临时文件（默认True）
    """
    print(f"正在下载模型到 {local_dir}...")
    
    # 创建目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 获取临时目录路径
    tmp_dir = os.environ.get('TMPDIR', '/tmp')
    
    try:
        # 下载并保存 processor
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(local_dir)
        print(f"Processor 已保存到 {local_dir}")
        
        # 下载并保存 model
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, 
            device_map="auto"
        )
        model.save_pretrained(local_dir)
        print(f"Model 已保存到 {local_dir}")
        
        # 清理临时文件
        if clean_tmp and os.path.exists(tmp_dir):
            print(f"清理临时目录: {tmp_dir}")
            import shutil
            # 只清理 HuggingFace 相关的临时文件
            for item in os.listdir(tmp_dir):
                if item.startswith('tmp') or 'huggingface' in item.lower():
                    tmp_path = os.path.join(tmp_dir, item)
                    try:
                        if os.path.isfile(tmp_path):
                            os.remove(tmp_path)
                        elif os.path.isdir(tmp_path):
                            shutil.rmtree(tmp_path)
                        print(f"  已删除: {tmp_path}")
                    except Exception as e:
                        print(f"  删除失败 {tmp_path}: {e}")
            print("临时文件清理完成")
        
        return processor, model
        
    except Exception as e:
        print(f"下载过程中出错: {e}")
        print(f"临时目录 {tmp_dir} 保留，请手动检查和清理")
        raise


def load_local_model(local_dir="./models/music_flamingo"):
    """
    从本地目录加载模型
    """
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"本地模型目录不存在: {local_dir}")
    
    print(f"从 {local_dir} 加载模型...")
    
    processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True, cache_dir='/network_space/server127_2/shared/sunyx3/huggingface')
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        local_dir, 
        device_map="auto",
        local_files_only=True,
        cache_dir='/network_space/server127_2/shared/sunyx3/huggingface'
    )
    
    print("模型加载完成！")
    return processor, model


# ===== 原有函数保持不变 =====

def summarize_song(segments, wav_path, processor, model):
    wav_name = os.path.basename(wav_path)
    m_index = wav_path.find('m')
    genre = extract_genre_from_filename(os.path.basename(wav_name)[m_index:m_index+3])
    
    overall_prompt = f"""Please briefly describe the overall mood and atmosphere of this {genre}-style music piece. 
    Keep your answer within 150 words. Make it brief and do not introduce the instrumentation and vocal arrangement of the song."""
    print(wav_path)
    overall_music_description = call_musicflamingo(overall_prompt, wav_path, processor, model)
    
    segmentation_info = f"This audio piece is divided into {len(segments)} segments.\n"
    for segment in segments:
        start_sec = segment['start_sec']
        end_sec = segment['end_sec']
        i = segments.index(segment)
        segmentation_info += f"Segment {i+1}: {start_sec:.2f} - {end_sec:.2f}.\n"
    print(segmentation_info)
    segmentation_prompt = (
        "Below is a manual segmentation of this song."
        f"{segmentation_info} "
        "Please provide an overall description of this music piece regarding the relationship between segments; "
        "and briefly describe the music in each segment."
    )

    segmentation_description = call_musicflamingo(segmentation_prompt, wav_path, processor, model)

    segments_description = ""
    for i in range(len(segments)):
        segment = segments[i]
        start_sec = segment['start_sec']
        end_sec = segment['end_sec']
        segment_summary = segment['sentence_summary']
        # 更新正则表达式，准确提取 "High-level movement synthesis" 部分
        pattern = r"High-level movement synthesis:\s*(.*?)(?=\s*Specific movement instructions for the model)"
        result = re.search(pattern, segment_summary, re.DOTALL)
        if result:
            extracted_text = result.group(1).strip()
            segments_description += f"Segment {i+1}: {start_sec:.2f} - {end_sec:.2f}.\n{extracted_text}\n"
    
    final_prompt = f"""
## Overall Musical Characteristics

{overall_music_description}


## Musical Structure and Segment Relationships

{segmentation_description}


## Choreographic Perspective and Design Intent

As a choreographer, my goal is to create a dance that responds to the music's overall emotional tone and structural organization. 
Each segment of movement should be shaped in accordance with the musical qualities, contrasts, and transitions described above, 
so that the choreography reflects both the global atmosphere of the piece and the local character of each segment.


## Segment-Level Choreographic Summaries

Below is the choreography I design in response to the musical structure described above.
Each segment corresponds exactly to the musical segment with the same index.
The movement content is intentionally shaped by the music's emotional tone, phrasing, and structural transitions.

{segments_description}
"""

    return final_prompt


def deal_folder(split_folder, out_folder, wav_folder, processor, model):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    for file_name in os.listdir(split_folder):
        out_path = os.path.join(out_folder, file_name)
        if os.path.exists(out_path):
            continue
        if not file_name.endswith('.json'):
            continue
            
        split_path = os.path.join(split_folder, file_name)
        wav_name = os.path.splitext(file_name)[0] + '.mp3'
        wav_path = os.path.join(wav_folder, wav_name)
        with open(split_path, 'r') as f:
            segments = json.load(f)
        
        large_summary = summarize_song(segments, wav_path, processor, model)
        print(large_summary)
        
        augmented_song = {
            'song_summary': large_summary,
            'segments': segments
        }
        
        with open(out_path, 'w') as f:
            json.dump(augmented_song, f, indent=4)
        print(f"Processed {file_name}, saved {len(segments)} segments to {out_path}")


if __name__ == "__main__":
    # ===== 重要：设置临时目录到有空间的位置 =====
    import tempfile
    # 将临时目录设置到你有足够空间的地方
    os.environ['HF_HOME'] = '/network_space/server127_2/shared/sunyx3/huggingface'
    os.environ['HF_HUB_CACHE'] = '/network_space/server127_2/shared/sunyx3/huggingface'
    
    os.environ['TMPDIR'] = '/network_space/server127_2/shared/sunyx3/tmp'
    os.environ['TEMP'] = '/network_space/server127_2/shared/sunyx3/tmp'
    os.environ['TMP'] = '/network_space/server127_2/shared/sunyx3/tmp'
    
    # 创建临时目录
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    split_folder = 'split/test'
    out_folder = 'song/test'
    wav_folder = '/network_space/server127_2/shared/caixhdata/aist_w/'
    local_model_dir = '/network_space/server127_2/shared/sunyx3/huggingface/hub/models--nvidia--music-flamingo-hf/snapshots/e29cfe92e682616f8f8014c60b2c5d17a37d4e33'
    
    # ===== 使用方式 =====
    
    # 方式1：首次运行，下载并保存模型（只需运行一次）
    # processor, model = download_and_save_model(local_dir=local_model_dir)
    
    # 方式2：后续运行，直接从本地加载（推荐）
    processor, model = load_local_model(local_dir=local_model_dir)
    
    # 处理数据
    deal_folder(split_folder, out_folder, wav_folder, processor, model)