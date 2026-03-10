import os
import json
import re
from pathlib import Path
from typing import List, Dict
import argparse
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ============= 配置参数 =============
# 请修改以下路径为实际路径
# JSON_FOLDER = "/network_space/server126/shared/sunyx/datasets/aist/text_description/song/test"  # 输入：包含.json文件的文件夹
JSON_FOLDER = "/network_space/server126/shared/sunyx/datasets/aist/text_description/finedance/song/test"
# AUDIO_FOLDER = "/network_space/server127_2/shared/caixhdata/aist_w/"  # 输入：包含.mp3文件的文件夹
AUDIO_FOLDER = "/network_space/server127/shared/sunyx/FineDance/data/finedance/music_mp3/test"
CHECKPOINT_DIR = "./final_model"  # 微调好的模型checkpoint路径
OUTPUT_DIR = "./finedance/choreo_song/test"  # 输出文件夹（包含.txt和.json）

# 设备选择：'cuda', 'cpu', 或 'auto'（自动选择）
DEVICE = "auto"  # 可选值: 'cuda', 'cpu', 'auto'

# 生成配置
MAX_NEW_TOKENS = 3072  # 最大生成token数（提高以减少截断）
TEMPERATURE = 0.7  # 温度参数，控制生成的随机性
TOP_P = 0.9  # nucleus sampling参数
DO_SAMPLE = True  # 是否使用采样
RETRY_ATTEMPTS = 3  # 不足时重试次数

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


def extract_genre(filename: str) -> str:
    """从文件名中提取舞蹈风格"""
    for i, char in enumerate(filename):
        if char == 'm' and i + 2 < len(filename):
            genre_code = 'm' + filename[i+1:i+3]
            if genre_code in GENRE_MAPPING:
                return GENRE_MAPPING[genre_code]
    return "Break"


def build_segmentation_text(segments: List[Dict]) -> str:
    """构建Segmentation文本"""
    seg_lines = []
    for i, seg in enumerate(segments, 1):
        start_sec = seg['start_sec']
        end_sec = seg['end_sec']
        seg_lines.append(f"Segment {i}: {start_sec:.2f} - {end_sec:.2f}")
    return "\n".join(seg_lines)


def build_prompt(segmentation: str, genre: str) -> str:
    """构建song-level prompt，与finetune.py DanceSongDataset._build_user_prompt 保持一致。
    
    训练时使用了 CoT 格式（<think>/<answer>），推理时同样使用该提示词，
    使模型输出 <think>...</think><answer>...</answer>，然后从 answer 块中解析结果。
    """
    prompt = f"""The audio is the background music for a {genre}-style dance.
It is divided into several segments:
{segmentation}
Please design the general choreography for each segment based on the overall characteristics of the piece and the structure of its segments.
For each segment, write at least one complete sentence describing the choreography.
Output the thinking process in <think> </think> and final answer in <answer> </answer>"""
    return prompt


# ============= [改动 1] 新增 helper：提取 <answer> 块，忽略 <think> =============
def extract_answer_only(text: str) -> str:
    """
    从模型输出中提取 <answer>...</answer> 块内的文本，忽略 <think>...</think>。

    为什么先提取 answer 再 parse：
    - 训练时模型输出格式为 <think>...</think><answer>...</answer>
    - <think> 块包含推理过程，不应被 parse 为编舞内容
    - 若直接对全文 parse，可能误提取 think 块中的示例文本
    - 因此必须先提取 answer 块，再对其做 segment/movement 解析

    行为：
    1. 优先匹配 <answer>...</answer>（大小写不敏感，跨行）
    2. 若匹配成功，返回 answer 内部文本（strip 后）
    3. 若没有 <answer>，fallback 返回原文（strip 后），并打印日志
    4. 清理残留标签（孤立 <think>/<answer> 等）
    """
    # 尝试提取 <answer> 块
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        result = answer_match.group(1).strip()
    else:
        print("⚠ No <answer> tag found, fallback to full text parsing")
        result = text.strip()

    # 清理残留标签（孤立 <think> / </think> / <answer> / </answer>）
    result = re.sub(r"</?think>|</?answer>", "", result, flags=re.IGNORECASE)
    return result.strip()


def parse_segments_from_text(generated_text: str, segments: List[Dict]) -> List[Dict]:
    """
    从生成的文本中解析出每个segment的编舞描述
    支持segment和text在同一行的情况
    
    Args:
        generated_text: 模型生成的完整文本（应传入 answer_text，即已提取 answer 块的文本）
        segments: 原始的segments列表（包含start_sec和end_sec）
    
    Returns:
        包含choreography、start_sec、end_sec的列表
    """
    result = []
    
    # 改进的正则表达式：匹配 "Segment X: 时间范围" 或 "Segment X"
    segment_pattern = r'Segment\s+(\d+)(?::\s*([\d.]+)\s*-\s*([\d.]+))?'
    
    # 找到所有segment标题的位置
    matches = list(re.finditer(segment_pattern, generated_text, re.IGNORECASE))
    
    if not matches:
        print("⚠ Warning: No segment patterns found in generated text")
        # 如果没有找到分段，将整个文本作为第一个segment
        if segments:
            result.append({
                'start_sec': segments[0]['start_sec'],
                'end_sec': segments[0]['end_sec'],
                'choreography': generated_text.strip()
            })
        return result
    
    # 提取每个segment的choreography
    for i, match in enumerate(matches):
        segment_num = int(match.group(1)) - 1  # 转换为0-based索引
        
        # 当前segment标题结束的位置
        header_end = match.end()
        
        # 下一个segment的开始位置
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(generated_text)
        
        # 提取choreography文本
        choreography_text = generated_text[header_end:next_start].strip()
        
        # 移除开头可能存在的冒号或其他分隔符
        choreography_text = re.sub(r'^[:：\-\s]+', '', choreography_text)
        
        # 使用原始segments中对应的时间信息
        if segment_num < len(segments):
            result.append({
                'start_sec': segments[segment_num]['start_sec'],
                'end_sec': segments[segment_num]['end_sec'],
                'choreography': choreography_text
            })
        else:
            print(f"⚠ Warning: Segment {segment_num + 1} found in text but index out of range")
    
    # 如果解析出的segment数量不匹配，给出警告
    if len(result) != len(segments):
        print(f"⚠ Warning: Parsed {len(result)} segments but expected {len(segments)}")
    
    return result


def format_output_text(segments: List[Dict]) -> str:
    """
    格式化输出文本，严格按照"segment 换行 text"的格式
    
    Args:
        segments: 包含choreography、start_sec、end_sec的列表
    
    Returns:
        格式化后的文本
    """
    def ensure_sentence(text: str) -> str:
        """确保每个segment至少有一句话，并补齐末尾标点。"""
        cleaned = text.strip()
        if not cleaned:
            return "The choreography continues with matching dynamics."
        if not re.search(r"[.!?]\s*$", cleaned):
            return cleaned + "."
        return cleaned

    lines = []
    
    for i, seg in enumerate(segments, 1):
        # Segment标题行
        segment_line = f"Segment {i}: {seg['start_sec']:.2f} - {seg['end_sec']:.2f}"
        lines.append(segment_line)
        
        # Choreography内容（换行）
        choreography = seg.get('choreography', '').strip()
        lines.append(ensure_sentence(choreography))
        
        # 段落之间添加空行（除了最后一个）
        if i < len(segments):
            lines.append("")
    
    return "\n".join(lines)


def load_model_and_processor(checkpoint_dir: str, device: torch.device):
    """
    从checkpoint加载模型和processor。
    
    [改动 6] 自动兼容 PEFT/LoRA adapter-only checkpoint：
    - 若 checkpoint_dir 下存在 adapter_config.json，视为 PEFT adapter 目录
    - 先从 adapter_config.json 读取 base_model_name_or_path，加载 base model
    - 再用 PeftModel.from_pretrained 挂载 adapter
    - 否则按原逻辑直接 from_pretrained
    """
    print(f"Loading model from: {checkpoint_dir}")

    # 加载processor（优先从 checkpoint_dir 加载）
    try:
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
        print(f"✓ Processor loaded from: {checkpoint_dir}")
    except Exception as e:
        print(f"⚠ Failed to load processor from checkpoint_dir: {e}")
        raise

    # 确保tokenizer有pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print("✓ Set pad_token to eos_token")

    # 根据设备类型选择数据类型
    if device.type == 'cpu':
        torch_dtype = torch.float32
        print("✓ Using torch.float32 for CPU")
    else:
        torch_dtype = torch.bfloat16
        print("✓ Using torch.bfloat16 for GPU")

    # 检测是否为 PEFT adapter-only checkpoint
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    is_peft_adapter = os.path.exists(adapter_config_path)

    if is_peft_adapter:
        print("✓ Detected PEFT adapter checkpoint, loading base model + adapter")
        # 读取 base model 路径
        with open(adapter_config_path, 'r', encoding='utf-8') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config.get("base_model_name_or_path", "")
        if not base_model_path:
            raise ValueError("adapter_config.json missing 'base_model_name_or_path'")
        print(f"  Base model path: {base_model_path}")

        # 加载 base model
        base_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            base_model_path,
            device_map={"": device},
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if device.type == 'cpu':
            base_model = base_model.float()

        # 挂载 PEFT adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        print("✓ PEFT adapter loaded successfully")
    else:
        # 非 adapter-only：直接加载
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            device_map={"": device},
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        if device.type == 'cpu':
            model = model.float()

    # 设置为评估模式
    model.eval()

    # 同步 pad_token_id 到 model.config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        print("✓ Set model.config.pad_token_id")

    print("✓ Model and processor loaded successfully")
    return model, processor


def generate_choreography(model, processor, prompt: str, audio_path: str, device: torch.device, max_new_tokens: int) -> str:
    """生成编舞描述"""
    # 构建对话格式
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_path},
            ],
        }
    ]
    
    # 处理输入
    inputs = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,  # 推理时需要生成prompt
        return_dict=True,
        return_tensors="pt",
    )
    
    # 获取模型的dtype
    model_dtype = next(model.parameters()).dtype
    
    # 移动到设备并转换dtype
    inputs = {
        k: v.to(device=device, dtype=model_dtype) if v.dtype in [torch.float32, torch.float16, torch.bfloat16] else v.to(device)
        for k, v in inputs.items()
    }
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # 解码（只解码新生成的token，去除输入部分）
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return generated_text


def is_generation_sufficient(parsed_segments: List[Dict], expected_segments: List[Dict]) -> bool:
    """判断生成是否足够：段数一致且每段至少有一句话（非空）。"""
    if len(parsed_segments) != len(expected_segments):
        return False
    for seg in parsed_segments:
        if not seg.get('choreography', '').strip():
            return False
    return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Song-Level Choreography Generation')
    parser.add_argument('--device', type=str, default=DEVICE, 
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device to use: cuda, cpu, or auto (default: auto)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Song-Level Choreography Generation")
    print(f"{'='*60}")
    print(f"JSON Folder: {JSON_FOLDER}")
    print(f"Audio Folder: {AUDIO_FOLDER}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # 验证路径
    if not os.path.exists(JSON_FOLDER):
        print(f"❌ Error: JSON_FOLDER does not exist: {JSON_FOLDER}")
        return
    if not os.path.exists(AUDIO_FOLDER):
        print(f"❌ Error: AUDIO_FOLDER does not exist: {AUDIO_FOLDER}")
        return
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Error: CHECKPOINT_DIR does not exist: {CHECKPOINT_DIR}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_output_dir = os.path.join(OUTPUT_DIR, "txt")
    json_output_dir = os.path.join(OUTPUT_DIR, "json")
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Auto-selected device: {device}")
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: {device}")
        else:
            print("⚠ Warning: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    else:  # cpu
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # 加载模型
    model, processor = load_model_and_processor(CHECKPOINT_DIR, device)
    
    # 获取所有json文件
    json_folder = Path(JSON_FOLDER)
    json_files = sorted(list(json_folder.glob("*.json")))
    print(f"Found {len(json_files)} JSON files\n")
    
    # 遍历处理每个文件
    for json_path in tqdm(json_files, desc="Processing files"):
        try:
            json_output_path = os.path.join(json_output_dir, f"{json_path.stem}.json")
            if os.path.exists(json_output_path):
                continue
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = data.get('segments', [])
            if not segments:
                print(f"\n⚠ Warning: No segments found in {json_path.name}, skipping")
                continue
            
            # 获取对应的音频文件
            audio_filename = json_path.stem + ".mp3"
            audio_path = Path(AUDIO_FOLDER) / audio_filename
            
            if not audio_path.exists():
                print(f"\n⚠ Warning: Audio file not found: {audio_path}, skipping")
                continue
            
            # 提取舞蹈风格
            genre = extract_genre(json_path.stem)
            
            # 构建segmentation文本
            segmentation = build_segmentation_text(segments)
            
            # 构建prompt（与 finetune 保持一致，包含 CoT 指令）
            prompt = build_prompt(segmentation, genre)
            
            # 生成编舞
            print(f"\n{'='*50}")
            print(f"Processing: {json_path.stem}")
            print(f"Genre: {genre}")
            print(f"Segments: {len(segments)}")
            print(f"Generating...")
            
            parsed_segments = []
            generated_text = ""
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                print(f"Attempt {attempt}/{RETRY_ATTEMPTS} with max_new_tokens={MAX_NEW_TOKENS}")
                generated_text = generate_choreography(
                    model, processor, prompt, str(audio_path), device, MAX_NEW_TOKENS
                )

                print(f"✓ Generation completed ({len(generated_text)} chars)")

                # [改动 2] 先提取 <answer> 块，再对其做 segment 解析
                # 原因：训练时输出格式为 <think>...</think><answer>...</answer>，
                # <think> 块中可能包含示例文本，若直接全文解析会误提取 think 中的 Segment 片段。
                answer_text = extract_answer_only(generated_text)

                # [改动 7] 用 answer_text 做 segment parsing（不含 thinking）
                parsed_segments = parse_segments_from_text(answer_text, segments)

                if is_generation_sufficient(parsed_segments, segments):
                    break
                print("⚠ Generation insufficient, retrying...")
            
            # 格式化输出文本（严格按照segment换行text的格式）
            formatted_text = format_output_text(parsed_segments)
            
            # 保存输出到.txt文件（含 debug 信息：prompt、音频路径、原始生成文本、格式化输出）
            txt_output_path = os.path.join(txt_output_dir, f"{json_path.stem}.txt")
            debug_header = (
                f"=== DEBUG INFO ===\n"
                f"Audio: {audio_path}\n"
                f"=== PROMPT ===\n"
                f"{prompt}\n"
                f"=== RAW GENERATED TEXT ===\n"
                f"{generated_text}\n"
                f"=== FORMATTED OUTPUT ===\n"
            )
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(debug_header + formatted_text)
            print(f"✓ Saved output to: {txt_output_path}")
            
            # 保存为JSON格式（只包含segments，输出 schema 不变）
            output_data = {
                "segments": parsed_segments
            }
            json_output_path = os.path.join(json_output_dir, f"{json_path.stem}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved parsed JSON to: {json_output_path}")
            print(f"✓ Parsed {len(parsed_segments)} segments")
            
        except Exception as e:
            print(f"\n❌ Error processing {json_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"All files processed!")
    print(f"TXT outputs: {txt_output_dir}")
    print(f"JSON outputs: {json_output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
