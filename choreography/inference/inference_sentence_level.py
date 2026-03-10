import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from pydub import AudioSegment

# ============= 配置参数 =============
JSON_FOLDER = "./finedance/choreo_song/test/json"  # 输入：song-level生成的json文件夹
FULL_AUDIO_DIR = "/network_space/server127/shared/sunyx/FineDance/data/finedance/music_mp3/test"
AUDIO_CACHE_DIR = "../audio_segments_cache"  # 音频片段缓存目录
CHECKPOINT_DIR = "./final_model"  # 微调好的模型checkpoint路径
OUTPUT_DIR = "./finedance/choreo/test/"  # 输出文件夹（包含.txt和.json）

# 设备选择：仅支持 CUDA
DEVICE = "cuda"

# 生成配置
MAX_NEW_TOKENS = 3072  # 最大生成token数
TEMPERATURE = 0.7  # 温度参数，控制生成的随机性
TOP_P = 0.9  # nucleus sampling参数
DO_SAMPLE = True  # 是否使用采样
MAX_RETRY = 5  # 解析失败时的最大重试次数


def extract_body_part(modifier_text: str, part: str = 'all') -> Tuple[str, bool]:
    """
    从modifier文本中提取指定部分

    Args:
        modifier_text: modifier文本字符串
        part: 要提取的部分 ('whole', 'upper', 'lower', 'torso', 'simple_tag', 'all')

    Returns:
        tuple: (提取的文本, 是否成功找到)
    """
    if part == 'all':
        return modifier_text, True

    patterns = {
        'whole': r'\*\*whole body\*\*:\s*([^\n*]+)',
        'upper': r'\*\*upper body\*\*:\s*([^\n*]+)',
        'lower': r'\*\*lower body\*\*:\s*([^\n*]+)',
        'torso': r'\*\*torso\*\*:\s*([^\n*]+)',
        'simple_tag': r'\*\*simple tag\*\*:\s*([^\n*]+)',
    }

    match = re.search(patterns[part], modifier_text, re.IGNORECASE)
    if match:
        extracted_text = match.group(1).strip()
        extracted_text = re.sub(r'^\*+\s*', '', extracted_text)
        return extracted_text, True
    else:
        return "", False


def cut_audio_segment(full_audio_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
    """
    从完整音频中切剆出指定时间段的片段
    
    Args:
        full_audio_path: 完整音频文件路径
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒）
        output_path: 输出文件路径
    
    Returns:
        bool: 是否成功切剆
    """
    try:
        # 加载完整音频
        audio = AudioSegment.from_file(full_audio_path)
        
        # 转换为毫秒
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # 切剆音频
        segment = audio[start_ms:end_ms]
        
        # 导出
        segment.export(output_path, format="mp3")
        return True
    except Exception as e:
        print(f"❌ Error cutting audio segment: {e}")
        return False


def build_prompt(summary: str) -> str:
    """
    构建sentence-level prompt，与 finetune.py DanceSegmentDataset.__getitem__ 中 prompt_segment 保持一致。
    
    训练时使用了 CoT 格式，推理时使用同样提示词，使模型输出 <think>...</think><answer>...</answer>。
    解析时先用 extract_answer_only() 提取 answer 块，再从中解析 movement 列表。
    """
    prompt = f"""Given a general choreographical instruction for this music segment, please develop it into a detailed motion sequence with precise timestamps and specific descriptions for different body parts.
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
    return prompt


# ============= [改动 1] 新增 helper：提取 <answer> 块，忽略 <think> =============
def extract_answer_only(text: str) -> str:
    """
    从模型输出中提取 <answer>...</answer> 块内的文本，忽略 <think>...</think>。

    为什么先提取 answer 再 parse：
    - 训练时模型输出格式为 <think>...</think><answer>...</answer>
    - <think> 块包含推理过程，不应被 parse 为 movement 内容
    - 若直接对全文 parse，可能误提取 think 块中的示例 movement 文本
    - 因此必须先提取 answer 块，再对其做 movement 解析

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


def parse_movements(generated_text: str) -> List[Dict]:
    """
    从生成的文本中解析出所有movements。
    应传入 answer_text（即已经过 extract_answer_only 提取的文本），而非原始模型输出。
    
    [改动 4] 同时支持两种 movement 标题格式：
    1. 训练格式（优先）：  * **Movement 1 (0.36 - 1.48):**
    2. 推理兼容格式：       ***Movement 1***: 0.00 – 1.40  /  **Movement 1**: 0.00 - 1.40
    
    [改动 5] 新增 simple tag 字段提取与校验，但不写入最终 JSON schema。

    Returns:
        包含start_sec、end_sec、start_frame、end_frame、modifier字典的列表
        modifier 结构不变：{'whole': ..., 'upper': ..., 'lower': ..., 'torso': ...}
    """
    movements = []

    # ---- 格式1：训练格式（优先匹配）----
    # 匹配: * **Movement 1 (0.36 - 1.48):**
    # 支持整数/小数时间，支持 - / – / — 三种连接符
    pattern_train = r'\*\s+\*{2,3}Movement\s+(\d+)\s*\((\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)\)\s*:{1,3}\*{0,3}'

    # ---- 格式2：推理兼容格式 ----
    # 匹配: ***Movement 1***: 0.00 – 1.40  /  **Movement 1**: 0.00 - 1.40
    pattern_compat = r'\*{2,3}Movement\s*(\d+)\*{2,3}:\s*(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)'

    matches_train = list(re.finditer(pattern_train, generated_text, re.IGNORECASE))
    matches_compat = list(re.finditer(pattern_compat, generated_text, re.IGNORECASE))

    # 优先使用训练格式；若未匹配到，fallback 到兼容格式
    if matches_train:
        matches = matches_train
    elif matches_compat:
        print("⚠ Training-format movement pattern not found, fallback to compat format")
        matches = matches_compat
    else:
        print("⚠ Warning: No movement patterns found")
        return movements

    # 限制最多7个movements
    if len(matches) > 7:
        print(f"⚠ Warning: Found {len(matches)} movements, limiting to first 7")
        matches = matches[:7]

    for i, match in enumerate(matches):
        # group(1)=编号, group(2)=start, group(3)=end
        start_sec = float(match.group(2))
        end_sec = float(match.group(3))

        # 提取这个 movement 的 modifier 文本
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(generated_text)
        modifier_text = generated_text[start_pos:end_pos].strip()

        # 提取各个body part
        simple_tag, found_simple_tag = extract_body_part(modifier_text, 'simple_tag')
        whole, found_whole = extract_body_part(modifier_text, 'whole')
        upper, found_upper = extract_body_part(modifier_text, 'upper')
        lower, found_lower = extract_body_part(modifier_text, 'lower')
        torso, found_torso = extract_body_part(modifier_text, 'torso')

        # 验证所有必需 body part 都存在（包括 simple tag）
        missing_parts = []
        if not found_whole:
            missing_parts.append('whole body')
        if not found_upper:
            missing_parts.append('upper body')
        if not found_lower:
            missing_parts.append('lower body')
        if not found_torso:
            missing_parts.append('torso')
        if not found_simple_tag:
            missing_parts.append('simple tag')

        if missing_parts:
            print(f"⚠ Warning: Movement {i+1} missing body parts: {', '.join(missing_parts)}")
            # 跳过这个不完整的movement
            continue

        # 计算frame
        start_frame = round(start_sec * 30)
        end_frame = round(end_sec * 30)

        # [改动 5] modifier 结构保持不变（whole/upper/lower/torso），不新增 simple_tag 字段
        movements.append({
            'start_sec': start_sec,
            'end_sec': end_sec,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'modifier': {
                'simple_tag': simple_tag,
                'whole_body': whole,
                'upper_body': upper,
                'lower_body': lower,
                'torso': torso
            }
        })

    return movements


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

    # 使用 bfloat16 for GPU
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

    # 设置为评估模式
    model.eval()

    # 同步 pad_token_id 到 model.config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        print("✓ Set model.config.pad_token_id")

    print("✓ Model and processor loaded successfully")
    return model, processor


def generate_movements(model, processor, prompt: str, audio_path: str, device: torch.device) -> str:
    """生成movement序列"""
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
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # 移动到设备并转换为bfloat16
    inputs = {k: v.to(device=device, dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to(device) 
              for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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


def main():
    """主函数"""
    # 关闭底层 fd=0（stdin），防止后台运行时 processor/ffmpeg 等 C 扩展子进程
    # 尝试读取终端 stdin，触发 SIGTTIN 信号导致整个进程组被挂起（Stopped）。
    # 仅替换 sys.stdin 不够，必须在 os 层关闭 fd=0，否则 C 扩展/子进程仍会继承原始 fd。
    import sys
    try:
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull_fd, 0)  # 将 fd=0 重定向到 /dev/null
        os.close(devnull_fd)
        sys.stdin = open(os.devnull, 'r')
    except Exception:
        pass

    # 解析命令行参数（保留参数接口以兼容旧脚本，但仅支持cuda）
    parser = argparse.ArgumentParser(description='Sentence-Level Movement Generation')
    parser.add_argument('--device', type=str, default=DEVICE, 
                        choices=['cuda'],
                        help='Device to use: cuda only (GPU required)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Sentence-Level Movement Generation")
    print(f"{'='*60}")
    print(f"JSON Folder: {JSON_FOLDER}")
    print(f"Full Audio Dir: {FULL_AUDIO_DIR}")
    print(f"Audio Cache Dir: {AUDIO_CACHE_DIR}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # 验证路径
    if not os.path.exists(JSON_FOLDER):
        print(f"❌ Error: JSON_FOLDER does not exist: {JSON_FOLDER}")
        return
    if not os.path.exists(FULL_AUDIO_DIR):
        print(f"❌ Error: FULL_AUDIO_DIR does not exist: {FULL_AUDIO_DIR}")
        return
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"❌ Error: CHECKPOINT_DIR does not exist: {CHECKPOINT_DIR}")
        return
    
    # 创建输出目录和缓存目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
    txt_output_dir = os.path.join(OUTPUT_DIR, "txt")
    json_output_dir = os.path.join(OUTPUT_DIR, "json")
    os.makedirs(txt_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # 设置设备（仅支持CUDA）
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # 加载模型
    model, processor = load_model_and_processor(CHECKPOINT_DIR, device)
    
    # 获取所有json文件
    json_folder = Path(JSON_FOLDER)
    json_files = sorted(list(json_folder.glob("*.json")))
    print(f"Found {len(json_files)} JSON files\n")
    
    # 统计信息
    total_sentences = 0
    processed_sentences = 0
    
    # 遍历处理每个文件
    for json_path in tqdm(json_files, desc="Processing files"):
        try:
            base_name = json_path.stem
            json_output_path = os.path.join(json_output_dir, f"{base_name}.json")
            if os.path.exists(json_output_path):
                print(f"\n⚠ Warning: Output JSON already exists for {json_path.name}, skipping")
                continue
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                sentences = json.load(f)
            sentences = sentences['segments']
            
            if not sentences:
                print(f"\n⚠ Warning: No sentences found in {json_path.name}, skipping")
                continue
            
            base_name = json_path.stem
            print(f"\n{'='*50}")
            print(f"Processing: {base_name}")
            print(f"Total sentences: {len(sentences)}")
            
            # 查找对应的完整音频文件
            full_audio_path = Path(FULL_AUDIO_DIR) / f"{base_name}.mp3"
            if not full_audio_path.exists():
                print(f"\n⚠ Warning: Full audio file not found: {full_audio_path}, skipping")
                continue
            
            # 准备输出数据
            output_data = []
            
            # 遍历每个sentence
            for idx, sentence in enumerate(sentences, 1):
                total_sentences += 1
                
                start_sec = sentence['start_sec']
                end_sec = sentence['end_sec']
                choreography = sentence['choreography']
                
                # 构建音频片段文件名
                audio_filename = f"{base_name}_{start_sec:.2f}_{end_sec:.2f}.mp3"
                audio_path = Path(AUDIO_CACHE_DIR) / audio_filename
                
                # 如果缓存中不存在，则切剆音频
                if not audio_path.exists():
                    print(f"  Cutting audio segment: {audio_filename}")
                    if not cut_audio_segment(str(full_audio_path), start_sec, end_sec, str(audio_path)):
                        print(f"\n⚠ Warning: Failed to cut audio segment, skipping")
                        continue
                
                # 构建prompt（与 finetune 保持一致，包含 CoT 指令）
                prompt = build_prompt(choreography)
                
                # 生成movements，带重试机制
                print(f"\n  Sentence {idx}/{len(sentences)} ({start_sec:.2f}-{end_sec:.2f}):")
                
                movements = []
                generated_text = ""
                answer_text = ""
                for retry_count in range(MAX_RETRY):
                    print(f"  Generating... (attempt {retry_count + 1}/{MAX_RETRY})")
                    
                    generated_text = generate_movements(
                        model, processor, prompt, str(audio_path), device
                    )
                    
                    print(f"  ✓ Generation completed ({len(generated_text)} chars)")

                    # [改动 2] 先提取 <answer> 块，再对其做 movement 解析
                    # 原因：训练时输出格式为 <think>...</think><answer>...</answer>，
                    # <think> 块中可能包含示例 movement 文本，若直接全文解析会产生误提取。
                    answer_text = extract_answer_only(generated_text)

                    # [改动 4] 用 answer_text 做 movement parsing
                    movements = parse_movements(answer_text)
                    
                    if len(movements) > 0:
                        print(f"  ✓ Parsed {len(movements)} movements")
                        break
                    else:
                        print(f"  ⚠ No movements parsed, retrying...")
                
                # 如果所有重试都失败，记录警告
                if len(movements) == 0:
                    print(f"  ❌ Failed to parse movements after {MAX_RETRY} attempts, skipping this sentence")
                    continue
                
                # [改动 8] 保存txt时写入 answer_text（不含 thinking）
                # 文件名/路径/数量保持不变，只是不再写入 thinking 内容
                txt_filename = f"{base_name}_sentence{idx}.txt"
                txt_output_path = os.path.join(txt_output_dir, txt_filename)
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write(answer_text)
                
                # 构建输出数据（JSON schema 不变）
                sentence_data = {
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'choreography': choreography,
                    'sentence': movements
                }
                output_data.append(sentence_data)
                
                processed_sentences += 1
            
            # 保存为JSON格式（整个文件的所有sentences，schema 不变）
            json_output_path = os.path.join(json_output_dir, f"{base_name}.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Saved JSON to: {json_output_path}")
            print(f"✓ Processed {processed_sentences}/{total_sentences} sentences for this file")
            
        except Exception as e:
            print(f"\n❌ Error processing {json_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"All files processed!")
    print(f"Total sentences processed: {processed_sentences}/{total_sentences}")
    print(f"TXT outputs: {txt_output_dir}")
    print(f"JSON outputs: {json_output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
