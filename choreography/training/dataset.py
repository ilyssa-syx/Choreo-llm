import json
import re
from pathlib import Path
from typing import List, Dict
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from tqdm import tqdm


# 舞蹈风格映射
GENRE_MAPPING = {
    'mBR': 'Break',
    'mPO': 'Pop',
    'mLO': 'Lock',
    'mWA': 'Waack',
    'mMH': 'Middle Hip-hop',
    'mLH': 'LA-style Hip-hop',
    'mHO': 'House',
    'mKR': 'Krump',
    'mJS': 'Street Jazz',
    'mJB': 'Ballet Jazz'
}

# 音频切片缓存配置
DEFAULT_SEGMENT_CACHE_DIR = "./audio_segments_cache"


class DanceSongDataset(Dataset):
    """舞蹈歌曲级别数据集（原有数据）"""
    
    def __init__(self, json_folder: str, audio_folder: str, processor, ablation: str = "FULL"):
        self.json_folder = Path(json_folder)
        self.audio_folder = Path(audio_folder)
        self.processor = processor
        self.ablation = ablation
        
        # 获取所有 .json 文件
        all_json_files = sorted(list(self.json_folder.glob("*.json")))
        self.json_files = []
        
        # 验证 JSON 文件有效性
        for json_path in all_json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证是否有必需的字段
            assert 'segments' in data and 'song_summary' in data, f"Missing required fields in {json_path}"
            self.json_files.append(json_path)
            
        
        print(f"Found {len(self.json_files)} valid JSON files (out of {len(all_json_files)} total)")
        
    def __len__(self):
        return len(self.json_files)
    
    def _extract_genre(self, filename: str) -> str:
        """从文件名中提取舞蹈风格"""
        # 文件名格式: gBR_sBM_cAll_d04_mBR0_ch02.json
        # 查找小写 'm' 并提取后面的两个字符
        for i, char in enumerate(filename):
            if char == 'm' and i + 2 < len(filename):
                genre_code = 'm' + filename[i+1:i+3]
                if genre_code in GENRE_MAPPING:
                    return GENRE_MAPPING[genre_code]
        # 如果没有找到，返回默认值
        return "Break"
    
    def _build_segmentation_text(self, segments: List[Dict]) -> str:
        """构建 Segmentation 文本"""
        seg_lines = []
        for i, seg in enumerate(segments, 1):
            start_sec = seg['start_sec']
            end_sec = seg['end_sec']
            seg_lines.append(f"Segment {i}: {start_sec:.2f} - {end_sec:.2f}")
        return "\n".join(seg_lines)
    
    def _build_user_prompt(self, segmentation: str, genre: str) -> str:
        """构建用户提示词"""
        prompt_song = f"""The audio is the background music for a {genre}-style dance.
        It is divided into several segments:
        {segmentation}
        Please strictly follow the given segmentation and design the general choreography for each segment based on the overall characteristics of the piece and the structure of its segments.
        Output the thinking process in <think> </think> and final answer in <answer> </answer>"""
        return prompt_song
    
    def _build_assistant_response(self, song_summary: str) -> str:
        """构建助手响应"""
        # 找到 "## Segment-Level Choreographic Summaries" 的位置
        marker = "## Segment-Level Choreographic Summaries"
        assert marker in song_summary, f"Expected marker '{marker}' not found in song_summary"
        
        parts = song_summary.split(marker, 1)
        thinking_part = parts[0].strip()
        answer_part = parts[1].strip()
            
            # 构建最终响应
        response = f"<think>\n{thinking_part}\nHence, the final answer is:\n</think>\n<answer>\n{answer_part}\n</answer>"
        
        return response
    
    def _remove_think_tags(self, response: str) -> str:
        """移除<think></think>标签及其内容"""
        import re
        # 使用正则表达式移除 <think>...</think> 部分
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return response.strip()
    
    def _remove_think_instruction(self, prompt: str) -> str:
        """移除提示词中要求输出think标签的指令"""
        # 移除 "Output the thinking process in <think> </think> and final answer in <answer> </answer>"
        prompt = prompt.replace("Output the thinking process in <think> </think> and final answer in <answer> </answer>", 
                               "Output your answer in <answer> </answer>")
        return prompt
    
    def __getitem__(self, idx):
        # 读取 JSON 文件
        json_path = self.json_files[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取对应的音频文件
        audio_filename = json_path.stem + ".mp3"
        audio_path = self.audio_folder / audio_filename
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 从文件名中提取舞蹈风格
        genre = self._extract_genre(json_path.stem)
        
        # 构建 Segmentation 文本
        segments = data['segments']
        segmentation = self._build_segmentation_text(segments)
        
        # 构建用户提示词
        user_prompt = self._build_user_prompt(segmentation, genre)
        
        # 构建助手响应
        song_summary = data['song_summary']
        assistant_response = self._build_assistant_response(song_summary)
        
        # Ablation: 移除 <think> 标签
        if self.ablation == "NOCOT":
            user_prompt = self._remove_think_instruction(user_prompt)
            assistant_response = self._remove_think_tags(assistant_response)
        
        # 构建对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
        ]
        
        debug_info = {
            'idx': idx,
            'json_path': str(json_path),
            'audio_path': str(audio_path),
            'genre': genre,
            'num_segments': len(segments),
            'segmentation': segmentation,
            'user_prompt': user_prompt,
            'assistant_response': assistant_response,
            'dataset_type': 'song'
        }
        
        return conversation, str(json_path.stem), "song", debug_info


class DanceSegmentDataset(Dataset):
    """舞蹈片段级别数据集（新数据）"""
    
    def __init__(self, json_folder: str, audio_folder: str, processor, 
                 cache_dir: str = DEFAULT_SEGMENT_CACHE_DIR, ablation: str = "FULL"):
        self.json_folder = Path(json_folder)
        self.audio_folder = Path(audio_folder)
        self.processor = processor
        self.cache_dir = Path(cache_dir)
        self.ablation = ablation
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有 .json 文件并加载所有segments
        self.json_files = sorted(list(self.json_folder.glob("*.json")))
        self.segments = []
        
        print(f"Loading segments from {len(self.json_files)} JSON files...")
        skipped_files = 0
        
        for json_path in self.json_files:
            
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data = data['segments']
                # 确保data是列表
                if isinstance(data, list):
                    for segment in data:
                        # 确保segment有必要的字段
                        if 'start_sec' in segment and 'end_sec' in segment and 'sentence_summary' in segment:
                            self.segments.append({
                                'json_path': json_path,
                                'segment': segment
                            })
                else:
                    print(f"⚠ Warning: Skipping {json_path.name} - not a list")
                    skipped_files += 1
                    
            
        
        if skipped_files > 0:
            print(f"⚠ Skipped {skipped_files} invalid files")
        print(f"Found {len(self.segments)} valid segments")
        
        # ⚡ 性能优化：预处理所有音频切片
        self._preprocess_audio_segments()
    
    def __len__(self):
        return len(self.segments)
    
    def _preprocess_audio_segments(self):
        """预处理所有音频切片（仅首次运行时执行）"""
        print("\n⚡ Checking audio segment cache...")
        
        # 统计需要处理的切片数量
        segments_to_process = []
        for idx, item in enumerate(self.segments):
            cache_filename = self._get_cache_filename(item['json_path'], item['segment'])
            cache_path = self.cache_dir / cache_filename
            
            if not cache_path.exists():
                segments_to_process.append((idx, item, cache_path))
        
        if not segments_to_process:
            print(f"✓ All {len(self.segments)} segments already cached")
            return
        
        print(f"Processing {len(segments_to_process)}/{len(self.segments)} segments...")
        
        # 批量处理音频切片（仅主进程）
        for idx, item, cache_path in tqdm(segments_to_process, desc="Preprocessing audio segments", position=0, leave=True):
            json_path = item['json_path']
            segment = item['segment']
            
            # 获取完整音频路径
            audio_filename = json_path.stem + ".mp3"
            audio_path = self.audio_folder / audio_filename
            
            if not audio_path.exists():
                print(f"\n⚠ Warning: Audio not found: {audio_path}")
                continue
            
            # 提取并保存音频切片
            try:
                start_sec = segment['start_sec']
                end_sec = segment['end_sec']
                
                # 加载完整音频
                y, sr = librosa.load(str(audio_path), sr=None)
                
                # 裁剪音频
                start_sample = int(start_sec * sr)
                end_sample = int(end_sec * sr)
                y_segment = y[start_sample:end_sample]
                
                # 保存到缓存
                sf.write(str(cache_path), y_segment, sr)
            except Exception as e:
                print(f"\n⚠ Error processing {audio_path} [{start_sec}-{end_sec}]: {e}")
        
        print(f"✓ Audio preprocessing completed. {len(segments_to_process)} new segments cached.\n")
    
    def _get_cache_filename(self, json_path: Path, segment: Dict) -> str:
        """生成缓存文件名"""
        start_sec = segment['start_sec']
        end_sec = segment['end_sec']
        # 格式: gBR_sBM_cAll_d04_mBR0_ch02_0.36_1.48.mp3
        return f"{json_path.stem}_{start_sec:.2f}_{end_sec:.2f}.mp3"
    
    def _extract_genre(self, filename: str) -> str:
        """从文件名中提取舞蹈风格"""
        for i, char in enumerate(filename):
            if char == 'm' and i + 2 < len(filename):
                genre_code = 'm' + filename[i+1:i+3]
                if genre_code in GENRE_MAPPING:
                    return GENRE_MAPPING[genre_code]
        return "Break"
    
    def _extract_high_level_synthesis(self, sentence_summary: str) -> str:
        """从sentence_summary中提取High-level movement synthesis部分"""
        # 查找 "High-level movement synthesis:" 和 "Specific movement instructions" 之间的内容
        pattern = r'High-level movement synthesis:(.*?)(?=Specific movement instructions|$)'
        match = re.search(pattern, sentence_summary, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # 如果没有找到标准格式，尝试其他可能的变体
        pattern2 = r'High-level movement synthesis:(.*?)(?=\n\nSpecific|$)'
        match2 = re.search(pattern2, sentence_summary, re.DOTALL | re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        
        # 如果都没找到，返回空字符串
        return ""
    
    def _build_assistant_response_segment(self, sentence_summary: str) -> str:
        """构建片段级别的助手响应"""
        # 1. 删除 High-level movement synthesis 部分
        thinking_part = sentence_summary
        
        # 删除 High-level movement synthesis 部分及其内容
        pattern1 = r'High-level movement synthesis:.*?(?=Specific movement instructions|$)'
        thinking_part = re.sub(pattern1, '', thinking_part, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. 查找并提取 "Specific movement instructions for the model" 之后的内容
        answer_pattern = r'Specific movement instructions for the model:(.*?)$'
        answer_match = re.search(answer_pattern, thinking_part, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            answer_part = answer_match.group(1).strip()
            # 删除 "Specific movement instructions for the model" 这部分
            thinking_part = re.sub(r'Specific movement instructions for the model:.*$', '', thinking_part, flags=re.DOTALL | re.IGNORECASE)
            thinking_part = thinking_part.strip()
            
            # 构建最终响应
            response = f"<think>\n{thinking_part}\nHence, the final answer is:\n</think>\n<answer>\n{answer_part}\n</answer>"
        else:
            # 如果没有找到 Specific movement instructions，整个作为thinking
            response = f"<think>\n{thinking_part.strip()}\n</think>"
        
        return response
    
    def _remove_think_tags(self, response: str) -> str:
        """移除<think></think>标签及其内容"""
        import re
        # 使用正则表达式移除 <think>...</think> 部分
        response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return response.strip()

    def __getitem__(self, idx):
        item = self.segments[idx]
        json_path = item['json_path']
        segment = item['segment']
        
        # 提取时间段和摘要
        start_sec = segment['start_sec']
        end_sec = segment['end_sec']
        sentence_summary = segment['sentence_summary']
        
        # ⚡ 直接使用预处理好的缓存文件（避免实时IO）
        cache_filename = self._get_cache_filename(json_path, segment)
        audio_segment_path = str(self.cache_dir / cache_filename)
        
        if not Path(audio_segment_path).exists():
            raise FileNotFoundError(f"Cached audio segment not found: {audio_segment_path}")
        
        # 提取 High-level movement synthesis 作为 summary
        summary = self._extract_high_level_synthesis(sentence_summary)
        
        # 构建用户提示词（注意：DanceSegmentDataset 本来就不要求输出 think 标签，所以无需修改）
        prompt_segment = f"""Given a general choreographical instruction for this music segment, please develop it into a detailed motion sequence with precise timestamps and specific descriptions for different body parts.
Consider the musical style, rhythmic feel, mood, and the musical development indicated by the chord progression when arranging motion and transitions. 

Please output your answer in the following format:

* **Movement 1 (0.36 - 1.48):**

  **simple tag**: ... (must be <10 words, action-focused and describe the main body movements, explicitly mention torso direction and include key arm/leg actions when relevant, e.g., "body turning right while left leg kicks to the side and arms swing forward")
  **whole body**: ... (must be <25 words)
  **upper body**: ... (must be <15 words, describe left and right arms within this field)
  **lower body**: ... (must be <15 words, describe left and right legs within this field)
  **torso**: ... (must be <7 words)
  
* **Movement 2 (1.67 - 2.90):**
  ......
  
* Repeat for other movements

The general choreographical instruction for this music segment is as follows:
{summary}
"""
        
        # 构建助手响应
        assistant_response = self._build_assistant_response_segment(sentence_summary)
        
        # Ablation: 移除 <think> 标签
        if self.ablation == "NOCOT":
            assistant_response = self._remove_think_tags(assistant_response)
        
        # 构建对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_segment},
                    {"type": "audio", "path": audio_segment_path},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
        ]
        
        sample_name = f"{json_path.stem}_seg_{start_sec:.2f}_{end_sec:.2f}"
        
        debug_info = {
            'idx': idx,
            'json_path': str(json_path),
            'audio_path': audio_segment_path,
            'segment_time': f"{start_sec:.2f} - {end_sec:.2f}",
            'summary': summary,
            'user_prompt': prompt_segment,
            'assistant_response': assistant_response,
            'dataset_type': 'segment'
        }
        
        return conversation, sample_name, "segment", debug_info


class DanceOnestageDataset(Dataset):
    """舞蹈一阶段数据集（直接从音乐生成详细动作，无需segmentation）"""
    
    def __init__(self, json_folder: str, audio_folder: str, processor, ablation: str = "FULL"):
        # Onestage 本身就没有 Chain of Thought，必须使用 NOCOT 模式
        assert ablation == "NOCOT", f"DanceOnestageDataset must use ablation='NOCOT', got '{ablation}'"
        
        self.json_folder = Path(json_folder)
        self.audio_folder = Path(audio_folder)
        self.processor = processor
        self.ablation = ablation
        
        # 获取所有 .json 文件
        all_json_files = sorted(list(self.json_folder.glob("*.json")))
        self.json_files = []
        
        # 验证 JSON 文件有效性
        for json_path in all_json_files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证是否有必需的字段
            if 'segments' in data:
                # 确保每个segment都有详细动作描述（sentence数组）
                valid = True
                for seg in data['segments']:
                    if 'sentence' not in seg or not isinstance(seg['sentence'], list):
                        valid = False
                        break
                if valid:
                    self.json_files.append(json_path)
        
        print(f"Found {len(self.json_files)} valid JSON files for onestage (out of {len(all_json_files)} total)")
    
    def __len__(self):
        return len(self.json_files)
    
    def _extract_genre(self, filename: str) -> str:
        """从文件名中提取舞蹈风格"""
        for i, char in enumerate(filename):
            if char == 'm' and i + 2 < len(filename):
                genre_code = 'm' + filename[i+1:i+3]
                if genre_code in GENRE_MAPPING:
                    return GENRE_MAPPING[genre_code]
        return "Break"
    
    def _build_user_prompt(self, genre: str) -> str:
        """构建用户提示词（一阶段：直接从音乐生成详细动作）"""
        prompt = f"""The audio is the background music for a {genre}-style dance.
Please listen to the music and design detailed choreography with precise timestamps and specific descriptions for different body parts.
Consider the musical style, rhythmic patterns, mood changes, and structural development when creating the movements.

Please output your answer in the following format:

* **Movement 1 (0.36 - 1.48):**

  **simple tag**: ... (must be <10 words, action-focused and describe the main body movements, explicitly mention torso direction and include key arm/leg actions when relevant, e.g., "body turning right while left leg kicks to the side and arms swing forward")
  **whole body**: ... (must be <25 words)
  **upper body**: ... (must be <15 words, describe left and right arms within this field)
  **lower body**: ... (must be <15 words, describe left and right legs within this field)
  **torso**: ... (must be <7 words)
  
* **Movement 2 (1.67 - 2.90):**
  ......
  
* Repeat for all movements throughout the music

Output your answer in <answer> </answer>"""
        return prompt
    
    def _format_movement(self, motion: Dict, start_sec: float, end_sec: float) -> str:
        """格式化单个动作为期待的输出格式"""
        modifier = motion.get('modifier', {})
        
        lines = []
        lines.append(f"  **simple tag**: {modifier.get('simple_tag', 'N/A')}")
        lines.append(f"  **whole body**: {modifier.get('whole_body', 'N/A')}")
        lines.append(f"  **upper body**: {modifier.get('upper_half_body', 'N/A')}")
        lines.append(f"  **lower body**: {modifier.get('lower_half_body', 'N/A')}")
        lines.append(f"  **torso**: {modifier.get('torso', 'N/A')}")
        
        return "\n".join(lines)
    
    def _build_assistant_response(self, segments: List[Dict]) -> str:
        """构建助手响应（从sentence数组提取所有详细动作）"""
        answer_lines = []
        movement_counter = 1
        
        for segment in segments:
            sentence_array = segment.get('sentence', [])
            
            for motion in sentence_array:
                # 计算时间（从帧转换为秒，假设30fps）
                start_frame = motion.get('start_frame', 0)
                end_frame = motion.get('end_frame', 0)
                start_sec = start_frame / 30.0
                end_sec = end_frame / 30.0
                
                # 格式化动作
                answer_lines.append(f"* **Movement {movement_counter} ({start_sec:.2f} - {end_sec:.2f}):**")
                answer_lines.append("")
                answer_lines.append(self._format_movement(motion, start_sec, end_sec))
                answer_lines.append("")
                
                movement_counter += 1
        
        # 构建最终响应（无thinking部分）
        answer_text = "\n".join(answer_lines)
        response = f"<answer>\n{answer_text}\n</answer>"
        
        return response
    
    def __getitem__(self, idx):
        # 读取 JSON 文件
        json_path = self.json_files[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取对应的音频文件
        audio_filename = json_path.stem + ".mp3"
        audio_path = self.audio_folder / audio_filename
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # 从文件名中提取舞蹈风格
        genre = self._extract_genre(json_path.stem)
        
        # 获取所有segments
        segments = data['segments']
        
        # 构建用户提示词（不包含segmentation信息）
        user_prompt = self._build_user_prompt(genre)
        
        # 构建助手响应（从sentence数组提取所有详细动作，无thinking）
        assistant_response = self._build_assistant_response(segments)
        
        # 注意：DanceOnestageDataset 本身就没有 thinking，所以 ablation 不影响它
        # 但为了代码一致性，仍保留 ablation 参数（即使在这里不做任何操作）
        
        # 构建对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
        ]
        
        debug_info = {
            'idx': idx,
            'json_path': str(json_path),
            'audio_path': str(audio_path),
            'genre': genre,
            'num_segments': len(segments),
            'user_prompt': user_prompt,
            'assistant_response': assistant_response,
            'dataset_type': 'onestage'
        }
        
        return conversation, str(json_path.stem), "onestage", debug_info
