import os
import json
from pathlib import Path
from typing import List, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler, Subset
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import librosa
import soundfile as sf
import tempfile
import re
import shutil
import glob
import argparse

# ============= 配置参数 =============
# 请修改以下路径为实际路径
JSON_FOLDER = "/network_space/server126/shared/sunyx/datasets/aist/text_description/song/train"  # JSON文件所在文件夹
AUDIO_FOLDER = "/network_space/server127_2/shared/caixhdata/aist_w/"  # MP3音频文件所在文件夹

MODEL_ID = "/network_space/server127_2/shared/sunyx3/huggingface/hub/models--nvidia--music-flamingo-hf/snapshots/e29cfe92e682616f8f8014c60b2c5d17a37d4e33"
BATCH_SIZE = 1  # 音频编码器很吃显存，单卡 batch=1 是防 OOM 的底线
# NUM_EPOCHS 移除，改为命令行参数

# ============= LoRA 和量化配置 =============
USE_LORA = True
LOAD_IN_4BIT = True
# 3090 建议使用 bf16 (Ampere架构支持)，比 fp16 更稳定
COMPUTE_DTYPE = torch.bfloat16
# QLoRA 通常需要比全量微调更大的学习率
LEARNING_RATE = 2e-4  # 基础学习率（应用于所有数据）
SEGMENT_LR_SCALE = 0.5  # Segment数据的Loss缩放因子（通过缩放Loss达到降低LR的效果，0.5 = 等效LR为1e-4）
GRADIENT_ACCUMULATION_STEPS = 16  # 累积16次梯度，等效Batch Size=16，使梯度更稳定

# ============= 学习率调度器配置 =============
USE_SCHEDULER = True
WARMUP_RATIO = 0.1  # 前10%的步数用于warmup，防止训练初期梯度爆炸
SCHEDULER_TYPE = "cosine"  # 使用cosine调度器，让后期Loss收敛更平滑

# 保存配置
# CHECKPOINT_DIR 和 FINAL_MODEL_DIR 改为命令行参数
CHECKPOINT_DIR = None
FINAL_MODEL_DIR = None
SAVE_EVERY_N_STEPS = 100

# 数据子采样配置
# 取值范围 (0, 1]，例如 0.2 表示只用 20% 数据训练
# DATA_FRACTION 改为命令行参数
DATA_FRACTION = None
# 子采样策略: "uniform" 等间隔抽样; "random" 随机抽样
DATA_SAMPLING_STRATEGY = "uniform"
DATA_FRACTION_SEED = 42  # 随机抽样时的固定随机种子

# 音频切片缓存配置（性能优化）
SEGMENT_CACHE_DIR = "./audio_segments_cache"  # 预处理的音频切片保存路径

# 断点恢复配置
RESUME_FROM_CHECKPOINT = True  # 是否自动从最新checkpoint恢复训练

# DEBUG配置
DEBUG_MODE = False  # 设置为True时会输出详细的数据集信息
DEBUG_NUM_SAMPLES = 5  # DEBUG模式下每个数据集输出前N条样本
DEBUG_SAVE_TO_FILE = True  # 是否将DEBUG信息保存到文件
DEBUG_OUTPUT_FILE = "./debug_dataset_info.txt"  # DEBUG信息保存路径
# 控制展示哪个级别："both" / "song" / "segment"
DEBUG_SHOW_LEVEL = "both"

# 需要跳过的文件列表（sentence_summary 中缺少 High-level movement synthesis）

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


def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0


def find_latest_checkpoint(checkpoint_dir: str):
    """查找最新的checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # 查找所有 epoch_* 目录
    epoch_dirs = list(checkpoint_path.glob("epoch_*"))
    if not epoch_dirs:
        return None
    
    # 提取epoch数字并排序
    epoch_nums = []
    for ep_dir in epoch_dirs:
        try:
            epoch_num = int(ep_dir.name.split('_')[-1])
            epoch_nums.append((epoch_num, ep_dir))
        except ValueError:
            continue
    
    if not epoch_nums:
        return None
    
    # 返回最大epoch数的目录
    epoch_nums.sort(key=lambda x: x[0], reverse=True)
    latest_checkpoint = epoch_nums[0][1]
    
    # 检查是否包含模型文件（adapter或完整模型）
    has_adapter = (latest_checkpoint / "adapter_config.json").exists()
    has_model = (latest_checkpoint / "config.json").exists() or (latest_checkpoint / "adapter_model.safetensors").exists()
    
    if has_adapter or has_model:
        return str(latest_checkpoint)
    else:
        return None


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir, rank, world_size):
    """保存完整的checkpoint（模型+训练状态）"""
    if not is_main_process(rank):
        return

    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")

    # 1. 删除旧的 epoch checkpoint
    old_epoch_checkpoints = [d for d in glob.glob(os.path.join(checkpoint_dir, "epoch_*")) 
                            if d != epoch_checkpoint_path]
    for old_ckpt in old_epoch_checkpoints:
        # 保留每10个epoch的checkpoint
        try:
            ckpt_name = os.path.basename(old_ckpt)
            if ckpt_name.startswith("epoch_"):
                try:
                    ckpt_epoch = int(ckpt_name.split("_")[-1])
                except Exception:
                    ckpt_epoch = -1
                if ckpt_epoch > 0 and ckpt_epoch % 10 == 0:
                    print(f"✓ Preserved checkpoint: {ckpt_name} (multiple of 10)")
                    continue
            shutil.rmtree(old_ckpt)
            print(f"✓ Removed old checkpoint: {ckpt_name}")
        except Exception as e:
            print(f"⚠ Warning: Failed to remove {old_ckpt}: {e}")

    # 2. 保存新的 checkpoint
    os.makedirs(epoch_checkpoint_path, exist_ok=True)

    # 保存模型权重
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(epoch_checkpoint_path)

    # 保存训练状态
    trainer_state = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # 保存scheduler状态（如果有）
    if scheduler is not None:
        trainer_state['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(trainer_state, os.path.join(epoch_checkpoint_path, "trainer_state.pt"))

    print(f"✓ Checkpoint saved: {epoch_checkpoint_path} (epoch={epoch+1}, step={global_step})")
    print()


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank, world_size):
    """加载checkpoint并恢复训练状态"""
    if not os.path.exists(checkpoint_path):
        return 0, 0  # 返回起始epoch和global_step
    
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    # 加载模型权重
    model_to_load = model.module if hasattr(model, 'module') else model
    
    # 如果是PEFT模型，需要特殊处理
    if hasattr(model_to_load, 'load_adapter'):
        # PEFT模型：加载adapter权重
        from peft import PeftModel
        try:
            adapter_path = checkpoint_path
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                # 使用from_pretrained加载adapter权重到已有的PEFT模型
                import safetensors
                from safetensors.torch import load_file
                
                # 直接加载adapter权重文件
                adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    adapter_weights = load_file(adapter_weights_path)
                    # 加载权重到模型
                    model_to_load.load_state_dict(adapter_weights, strict=False)
                    print("✓ LoRA adapter weights loaded")
                else:
                    print("⚠ Warning: adapter_model.safetensors not found")
            else:
                print("⚠ Warning: No adapter_config.json found, skipping adapter loading")
        except Exception as e:
            print(f"⚠ Warning: Failed to load adapter: {e}")
            import traceback
            traceback.print_exc()
    
    # 尝试从目录名提取epoch信息（格式: epoch_7）
    start_epoch = 0
    global_step = 0
    
    try:
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name.startswith('epoch_'):
            start_epoch = int(checkpoint_name.split('_')[-1])
            print(f"✓ Extracted epoch {start_epoch} from checkpoint directory name")
    except Exception as e:
        print(f"⚠ Warning: Could not extract epoch from directory name: {e}")
    
    # 加载训练状态（如果存在）
    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
    if os.path.exists(trainer_state_path):
        trainer_state = torch.load(trainer_state_path, map_location='cpu')
        
        start_epoch = trainer_state.get('epoch', start_epoch)
        global_step = trainer_state.get('global_step', 0)
        
        # 恢复optimizer状态
        if 'optimizer_state_dict' in trainer_state:
            try:
                optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
                print(f"✓ Optimizer state restored")
            except Exception as e:
                print(f"⚠ Warning: Failed to load optimizer state: {e}")
        
        # 恢复scheduler状态
        if scheduler is not None and 'scheduler_state_dict' in trainer_state:
            try:
                scheduler.load_state_dict(trainer_state['scheduler_state_dict'])
                print(f"✓ Scheduler state restored")
            except Exception as e:
                print(f"⚠ Warning: Failed to load scheduler state: {e}")
        
        print(f"✓ Training state restored from trainer_state.pt")
    else:
        print("⚠ Warning: trainer_state.pt not found")
        print("⚠ Optimizer and scheduler states will be reset")
        print("⚠ Only model weights and epoch number will be restored")
    
    print(f"✓ Resuming from epoch {start_epoch}, global step {global_step}")
    print(f"{'='*60}\n")
    
    return start_epoch, global_step


def collate_fn(batch):
    """简单的collate函数，直接返回第一个元素（因为batch_size=1）"""
    if len(batch[0]) == 4:
        return batch[0][0], batch[0][1], batch[0][2], batch[0][3]
    elif len(batch[0]) == 3:
        return batch[0][0], batch[0][1], batch[0][2], None
    else:
        return batch[0][0], batch[0][1], "unknown", None


class DanceSongDataset(Dataset):
    """舞蹈歌曲级别数据集（原有数据）"""
    
    def __init__(self, json_folder: str, audio_folder: str, processor):
        self.json_folder = Path(json_folder)
        self.audio_folder = Path(audio_folder)
        self.processor = processor
        
        # 获取所有 .json 文件
        all_json_files = sorted(list(self.json_folder.glob("*.json")))
        self.json_files = []
        
        # 验证 JSON 文件有效性
        for json_path in all_json_files:
            # 跳过指定的文件
            if json_path.name in SKIP_FILES:
                print(f"⚠ Skipping {json_path.name} - in skip list")
                continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 验证是否有必需的字段
                if 'segments' in data and 'song_summary' in data:
                    self.json_files.append(json_path)
                else:
                    print(f"⚠ Warning: Skipping {json_path.name} - missing required fields")
            except json.JSONDecodeError as e:
                print(f"⚠ Warning: Skipping {json_path.name} - invalid JSON: {e}")
            except Exception as e:
                print(f"⚠ Warning: Skipping {json_path.name} - error: {e}")
        
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
Please design the general choreography for each segment based on the overall characteristics of the piece and the structure of its segments.
Output the thinking process in <think> </think> and final answer in <answer> </answer>"""
        return prompt_song
    
    def _build_assistant_response(self, song_summary: str) -> str:
        """构建助手响应"""
        # 找到 "## Segment-Level Choreographic Summaries" 的位置
        marker = "## Segment-Level Choreographic Summaries"
        
        if marker in song_summary:
            # 分割为 thinking 部分和 answer 部分
            parts = song_summary.split(marker, 1)
            thinking_part = parts[0].strip()
            answer_part = parts[1].strip()
            
            # 构建最终响应
            response = f"<think>\n{thinking_part}\nHence, the final answer is:\n</think>\n<answer>\n{answer_part}\n</answer>"
        else:
            # 如果没有找到标记，整个内容作为 thinking
            response = f"<think>\n{song_summary.strip()}\n</think>"
        
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
        
        # 构建 Segmentation 文本
        segments = data['segments']
        segmentation = self._build_segmentation_text(segments)
        
        # 构建用户提示词
        user_prompt = self._build_user_prompt(segmentation, genre)
        
        # 构建助手响应
        song_summary = data['song_summary']
        assistant_response = self._build_assistant_response(song_summary)
        
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
    
    def __init__(self, json_folder: str, audio_folder: str, processor, cache_dir: str = SEGMENT_CACHE_DIR, rank: int = 0, world_size: int = 1):
        self.json_folder = Path(json_folder)
        self.audio_folder = Path(audio_folder)
        self.processor = processor
        self.cache_dir = Path(cache_dir)
        self.rank = rank
        self.world_size = world_size
        
        # ⚠️ 修复竞态条件：只在主进程创建缓存目录
        if is_main_process(rank):
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 同步所有进程，确保目录已创建
        if world_size > 1:
            dist.barrier()
        
        # 获取所有 .json 文件并加载所有segments
        self.json_files = sorted(list(self.json_folder.glob("*.json")))
        self.segments = []
        
        print(f"Loading segments from {len(self.json_files)} JSON files...")
        skipped_files = 0
        
        for json_path in self.json_files:
            # 跳过指定的文件
            if json_path.name in SKIP_FILES:
                print(f"⚠ Skipping {json_path.name} - in skip list")
                skipped_files += 1
                continue
            
            try:
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
                    
            except json.JSONDecodeError as e:
                print(f"⚠ Warning: Skipping {json_path.name} - invalid JSON: {e}")
                skipped_files += 1
            except Exception as e:
                print(f"⚠ Warning: Skipping {json_path.name} - error: {e}")
                skipped_files += 1
        
        if skipped_files > 0:
            print(f"⚠ Skipped {skipped_files} invalid files")
        print(f"Found {len(self.segments)} valid segments")
        
        # ⚡ 性能优化：预处理所有音频切片（避免DDP竞态条件）
        self._preprocess_audio_segments()
        
        # 同步所有进程，确保预处理完成
        if world_size > 1:
            dist.barrier()
    
    def __len__(self):
        return len(self.segments)
    
    def _preprocess_audio_segments(self):
        """预处理所有音频切片（仅首次运行时执行）"""
        # ⚠️ 修复竞态条件：只在主进程进行预处理，其他进程等待
        if not is_main_process(self.rank):
            if self.world_size > 1:
                # 非主进程等待主进程完成预处理
                print(f"[Rank {self.rank}] Waiting for main process to finish preprocessing...")
            return
        
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
        
        # 构建用户提示词
        prompt_segment = f"""Given a general choreographical instruction for this music segment, please develop it into a detailed motion sequence with precise timestamps and specific descriptions for different body parts.
Consider the musical style, rhythmic feel, mood (as implied by the key), and the musical development indicated by the chord progression when arranging motion and transitions. 

Please output your answer in the following format:

* **Movement 1 (0.36 - 1.48):**

  **whole body**: ... (must be <25 words)
  **upper body**: ... (must be <15 words, describe left and right arms within this field)
  **lower body**: ... (must be <15 words, describe left and right legs within this field)
  **torso**: ... (must be <7 words)
  **simple tag**: ... (must be <10 words, action-focused and describe the main body movements, explicitly mention torso direction and include key arm/leg actions when relevant, e.g., "body turning right while left leg kicks to the side and arms swing forward")

* **Movement 2 (1.67 - 2.90):**
  ......
  
* Repeat for other movements

The general choreographical instruction for this music segment is as follows:
{summary}
"""
        
        # 构建助手响应
        assistant_response = self._build_assistant_response_segment(sentence_summary)
        
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


def train(num_epochs, data_fraction=None, checkpoint_dir=None, final_model_dir=None):
    """训练函数"""
    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"Distributed Training Setup")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Using device: {device}")
        print(f"Loading model: {MODEL_ID}")
        print(f"{'='*60}\n")
    
    # 1. 量化配置 (核心)
    bnb_config = None
    if LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # 二次量化，进一步节省显存
            bnb_4bit_quant_type="nf4",       # 4-bit NormalFloat，性能最好
            bnb_4bit_compute_dtype=COMPUTE_DTYPE
        )
        if is_main_process(rank):
            print("✓ 4-bit quantization enabled")
    
    # 2. 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 确保 Tokenizer 有 pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        if is_main_process(rank):
            print("✓ Set pad_token to eos_token")
    
    # 3. 加载模型
    try:
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": local_rank},  # 指定到具体的GPU
            trust_remote_code=True,
            # 如果支持 flash attention，使用它来加速和节省显存
            attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
        )
        if is_main_process(rank):
            print("✓ Model loaded successfully")
    except Exception as e:
        if is_main_process(rank):
            print(f"⚠ Failed to load with flash_attention_2, falling back to eager: {e}")
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": local_rank},  # 指定到具体的GPU
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    # 设置 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # ⚠️ 修复DDP兼容性：关闭cache，启用输入梯度
    model.config.use_cache = False
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    if is_main_process(rank):
        print("✓ DDP compatibility settings applied (use_cache=False, input_require_grads=True)")
    
    # 4. 启用梯度检查点 (极其重要！节省大量显存)
    # 使用 use_reentrant=False（PyTorch 推荐的新实现，更安全且与 DDP 兼容性更好）
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if is_main_process(rank):
        print("✓ Gradient checkpointing enabled (use_reentrant=False)")
    
    # 5. LoRA 配置和应用
    if USE_LORA:
        if is_main_process(rank):
            print("\nPreparing LoRA...")
        # 准备 k-bit 训练 (冻结原模型参数，norm 层转回 fp32)
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # AudioFlamingo 的语言模型部分通常是 Llama/Mistral 架构
            # 如果报错找不到层，可以先 print(model) 查看具体层名
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        if is_main_process(rank):
            model.print_trainable_parameters()
            print("✓ LoRA applied")
    
    # 6. 使用 DDP 包装模型
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=True,
            # ⚠️ 关键：允许 Gradient Checkpointing 重复使用参数
            # 模型结构在训练期间不会改变，使用静态图优化
            static_graph=True
        )
        if is_main_process(rank):
            print(f"✓ Model wrapped with DDP (world_size={world_size}, static_graph=True)")
    
    model.train()
    
    # 创建数据集
    if is_main_process(rank):
        print("\nCreating datasets...")
    # 原有的歌曲级别数据集
    song_dataset = DanceSongDataset(JSON_FOLDER, AUDIO_FOLDER, processor)
    if is_main_process(rank):
        print(f"Song-level dataset size: {len(song_dataset)}")
    
    # 新的片段级别数据集
    segment_dataset = DanceSegmentDataset(JSON_FOLDER, AUDIO_FOLDER, processor, rank=rank, world_size=world_size)
    if is_main_process(rank):
        print(f"Segment-level dataset size: {len(segment_dataset)}")
    
    # 合并两个数据集
    dataset = ConcatDataset([song_dataset, segment_dataset])
    # 可选：按比例子采样数据
    fraction = data_fraction if data_fraction is not None else DATA_FRACTION
    if fraction is not None and 0 < fraction < 1.0:
        total_size = len(dataset)
        subset_size = max(1, int(total_size * fraction))
        if DATA_SAMPLING_STRATEGY == "uniform":
            step = total_size / subset_size
            indices = [int(i * step) for i in range(subset_size)]
        else:
            g = torch.Generator()
            g.manual_seed(DATA_FRACTION_SEED)
            indices = torch.randperm(total_size, generator=g)[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        if is_main_process(rank):
            print(
                f"Using data fraction: {fraction} ({subset_size}/{total_size}), "
                f"strategy={DATA_SAMPLING_STRATEGY}"
            )
    if is_main_process(rank):
        print(f"Total dataset size: {len(dataset)}")
    
    # DEBUG: 输出数据集样本信息
    if DEBUG_MODE and is_main_process(rank):
        print("\n" + "="*80)
        print("DEBUG MODE: Inspecting dataset samples")
        print(f"Showing level: {DEBUG_SHOW_LEVEL}")
        print("="*80)
        
        debug_output_lines = []
        debug_output_lines.append("\n" + "="*80 + "\n")
        debug_output_lines.append("DEBUG MODE: Dataset Sample Inspection\n")
        debug_output_lines.append("="*80 + "\n")
        debug_output_lines.append(f"Generated at: {__import__('datetime').datetime.now()}\n")
        debug_output_lines.append(f"Song-level dataset size: {len(song_dataset)}\n")
        debug_output_lines.append(f"Segment-level dataset size: {len(segment_dataset)}\n")
        debug_output_lines.append(f"Total dataset size: {len(dataset)}\n\n")
        
        # Segment 样本（优先展示）
        if DEBUG_SHOW_LEVEL in ("both", "segment"):
            debug_output_lines.append("\n" + "="*80 + "\n")
            debug_output_lines.append("SEGMENT-LEVEL DATASET SAMPLES\n")
            debug_output_lines.append("="*80 + "\n")
            for i in range(min(DEBUG_NUM_SAMPLES, len(segment_dataset))):
                try:
                    conversation, sample_name, dataset_type, debug_info = segment_dataset[i]
                    print(f"\n{'='*80}")
                    print(f"Segment Sample {i+1}/{min(DEBUG_NUM_SAMPLES, len(segment_dataset))}")
                    print(f"{'='*80}")
                    print(f"JSON文件: {debug_info['json_path']}")
                    print(f"音频切片: {debug_info['audio_path']}")
                    print(f"时间范围: {debug_info['segment_time']}")
                    print(f"\n{'-'*80}")
                    print("Summary (High-level):")
                    print(f"{'-'*80}")
                    print(debug_info['summary'][:300] + "..." if len(debug_info['summary']) > 300 else debug_info['summary'])
                    print(f"\n{'-'*80}")
                    print("用户提示词 (User Prompt):")
                    print(f"{'-'*80}")
                    print(debug_info['user_prompt'][:500] + "..." if len(debug_info['user_prompt']) > 500 else debug_info['user_prompt'])
                    print(f"\n{'-'*80}")
                    print("助手响应 (Assistant Response - CoT):")
                    print(f"{'-'*80}")
                    print(debug_info['assistant_response'][:500] + "..." if len(debug_info['assistant_response']) > 500 else debug_info['assistant_response'])
                    
                    debug_output_lines.append(f"\n{'='*80}\n")
                    debug_output_lines.append(f"Segment Sample {i+1}\n")
                    debug_output_lines.append(f"JSON文件: {debug_info['json_path']}\n")
                    debug_output_lines.append(f"音频切片: {debug_info['audio_path']}\n")
                    debug_output_lines.append(f"时间范围: {debug_info['segment_time']}\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("Summary (High-level):\n")
                    debug_output_lines.append(debug_info['summary'] + "\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("用户提示词 (User Prompt):\n")
                    debug_output_lines.append(debug_info['user_prompt'] + "\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("助手响应 (Assistant Response - CoT):\n")
                    debug_output_lines.append(debug_info['assistant_response'] + "\n\n")
                except Exception as e:
                    error_msg = f"\n❌ Error loading segment sample {i}: {e}\n"
                    print(error_msg)
                    debug_output_lines.append(error_msg)
        
        # Song 样本
        if DEBUG_SHOW_LEVEL in ("both", "song"):
            debug_output_lines.append("\n" + "="*80 + "\n")
            debug_output_lines.append("SONG-LEVEL DATASET SAMPLES\n")
            debug_output_lines.append("="*80 + "\n")
            for i in range(min(DEBUG_NUM_SAMPLES, len(song_dataset))):
                try:
                    conversation, sample_name, dataset_type, debug_info = song_dataset[i]
                    print(f"\n{'='*80}")
                    print(f"Song Sample {i+1}/{min(DEBUG_NUM_SAMPLES, len(song_dataset))}")
                    print(f"{'='*80}")
                    print(f"JSON文件: {debug_info['json_path']}")
                    print(f"音频文件: {debug_info['audio_path']}")
                    print(f"舞蹈风格: {debug_info['genre']}")
                    print(f"片段数量: {debug_info['num_segments']}")
                    print(f"\n{'-'*80}")
                    print("用户提示词 (User Prompt):")
                    print(f"{'-'*80}")
                    print(debug_info['user_prompt'][:500] + "..." if len(debug_info['user_prompt']) > 500 else debug_info['user_prompt'])
                    print(f"\n{'-'*80}")
                    print("助手响应 (Assistant Response - CoT):")
                    print(f"{'-'*80}")
                    print(debug_info['assistant_response'][:500] + "..." if len(debug_info['assistant_response']) > 500 else debug_info['assistant_response'])
                    
                    debug_output_lines.append(f"\n{'='*80}\n")
                    debug_output_lines.append(f"Song Sample {i+1}\n")
                    debug_output_lines.append(f"JSON文件: {debug_info['json_path']}\n")
                    debug_output_lines.append(f"音频文件: {debug_info['audio_path']}\n")
                    debug_output_lines.append(f"舞蹈风格: {debug_info['genre']}\n")
                    debug_output_lines.append(f"片段数量: {debug_info['num_segments']}\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("用户提示词 (User Prompt):\n")
                    debug_output_lines.append(debug_info['user_prompt'] + "\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("助手响应 (Assistant Response - CoT):\n")
                    debug_output_lines.append(debug_info['assistant_response'] + "\n\n")
                except Exception as e:
                    error_msg = f"\n❌ Error loading song sample {i}: {e}\n"
                    print(error_msg)
                    debug_output_lines.append(error_msg)
        
        if DEBUG_SAVE_TO_FILE:
            try:
                with open(DEBUG_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.writelines(debug_output_lines)
                print(f"\n✓ DEBUG信息已保存到: {DEBUG_OUTPUT_FILE}")
            except Exception as e:
                print(f"\n⚠ 无法保存DEBUG文件: {e}")
        
        print("\n" + "="*80)
        print("DEBUG模式检查完成，请仔细查看以上输出")
        print("如果需要继续训练，请将 DEBUG_MODE 设置为 False")
        print("="*80 + "\n")
        
        response = input("\n是否继续训练? (y/n): ")
        if response.lower() != 'y':
            print("\n已取消训练，请检查数据集后再试。")
            cleanup_distributed()
            return
    
    # ⚠️ 关键：使用 DistributedSampler 进行多卡数据分配
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # 每个epoch随机打乱数据
        seed=42
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),  # 单卡时才使用shuffle
        collate_fn=collate_fn,
        num_workers=4,  # 多卡时减少worker数量避免冲突
        pin_memory=True
    )
    
    # 7. 使用 Paged AdamW 优化器 (显存不足时自动转移到内存)
    optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=LEARNING_RATE)
    if is_main_process(rank):
        print(f"✓ Optimizer created with lr={LEARNING_RATE}")
    
    # 8. 创建学习率调度器
    scheduler = None
    if USE_SCHEDULER:
        from transformers import get_cosine_schedule_with_warmup
        # 计算总训练步数（考虑分布式训练）
        num_training_steps = ((len(dataset) // world_size) // GRADIENT_ACCUMULATION_STEPS) * num_epochs
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        if is_main_process(rank):
            print(f"✓ Cosine scheduler created (warmup steps: {num_warmup_steps}/{num_training_steps})")
    
    # 创建检查点目录（只在主进程）
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
    final_dir = final_model_dir if final_model_dir is not None else FINAL_MODEL_DIR
    if is_main_process(rank):
        os.makedirs(ckpt_dir, exist_ok=True)
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # ⚡ 检查并加载最新checkpoint
    start_epoch = 0
    global_step = 0
    
    if RESUME_FROM_CHECKPOINT:
        latest_checkpoint = find_latest_checkpoint(ckpt_dir)
        if latest_checkpoint:
            start_epoch, global_step = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, rank, world_size
            )
        elif is_main_process(rank):
            print("\n✓ No checkpoint found, starting training from scratch\n")
    
    # 训练循环
    if is_main_process(rank):
        if start_epoch > 0:
            print(f"\n{'='*60}")
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Remaining epochs: {num_epochs - start_epoch}")
            print(f"{'='*60}\n")
        else:
            print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        if is_main_process(rank):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
        
        # 设置 epoch（用于 DistributedSampler 的 shuffle）
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        total_loss = 0
        optimizer.zero_grad()
        
        # 使用 tqdm 显示进度（只在主进程）
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}", disable=not is_main_process(rank))
        
        for idx, batch_data in enumerate(pbar):
            if len(batch_data) == 4:
                conversation, sample_name, dataset_type, _ = batch_data
            else:
                conversation, sample_name, dataset_type = batch_data[0], batch_data[1], batch_data[2]
            # 处理输入
            inputs = processor.apply_chat_template(
                [conversation],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                output_labels=True,
            )
            
            # 移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            
            # ⚠️ 关键：使用 Loss Scaling 而非修改 LR
            # Segment 数据的 loss 乘以 0.5，等效于 LR 降低一半
            # 这样做的好处：
            #   1. 不会破坏 Scheduler 的学习率衰减
            #   2. 与梯度累积完美兼容（16个样本可能包含Song和Segment混合）
            if dataset_type == "segment":
                loss = loss * SEGMENT_LR_SCALE
            
            # 梯度累积
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 梯度累积后更新参数
            if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # 更新进度条（只在主进程）
            if is_main_process(rank):
                avg_loss = total_loss / (idx + 1)
                # 获取当前实际学习率（来自scheduler）
                current_lr = optimizer.param_groups[0]['lr']
                # 计算等效学习率（对于segment数据，实际梯度大小等于lr*scale）
                effective_lr = current_lr * SEGMENT_LR_SCALE if dataset_type == "segment" else current_lr
                
                postfix_dict = {
                    'loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'eff_lr': f'{effective_lr:.2e}',  # 等效学习率
                    'type': dataset_type,
                    'sample': sample_name[:30],  # 截断过长的sample名称
                    'rank': rank
                }
                pbar.set_postfix(postfix_dict)
        
        # Epoch 结束后的统计（只在主进程）
        if is_main_process(rank):
            avg_epoch_loss = total_loss / len(dataloader)
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1} completed")
            print(f"Average Loss: {avg_epoch_loss:.4f}")
            print(f"{'='*50}")
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()
        
        # 每个 epoch 结束后保存检查点（只在主进程）
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, ckpt_dir, rank, world_size)
        # 保存processor（只需要保存一次，放在第一个epoch）
        if is_main_process(rank) and epoch == start_epoch:
            processor.save_pretrained(os.path.join(ckpt_dir, f"epoch_{epoch + 1}"))
            print(f"✓ Processor saved")
    
    # 保存最终模型（只在主进程）
    if is_main_process(rank):
        print(f"\n{'='*50}")
        print("Training completed! Saving final model...")
        os.makedirs(final_dir, exist_ok=True)
        # 保存时需要获取原始模型（去除DDP包装）
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
        print(f"✓ Final model saved to: {final_dir}")
        print(f"{'='*50}")
    
    # 清理分布式环境
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--data_fraction", type=float, default=None, help="训练数据比例 (0,1]，如0.2表示20%")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="checkpoint保存目录")
    parser.add_argument("--final_model_dir", type=str, default=None, help="最终模型保存目录")
    args = parser.parse_args()
    os.environ['HF_HOME'] = '/network_space/server127_2/shared/sunyx3/huggingface'
    os.environ['HF_HUB_CACHE'] = '/network_space/server127_2/shared/sunyx3/huggingface'
    os.environ['TMPDIR'] = '/network_space/server127_2/shared/sunyx3/tmp'
    os.environ['TEMP'] = '/network_space/server127_2/shared/sunyx3/tmp'
    os.environ['TMP'] = '/network_space/server127_2/shared/sunyx3/tmp'
    # 检查路径是否设置
    if JSON_FOLDER == "/path/to/json/folder" or AUDIO_FOLDER == "/path/to/audio/folder":
        print("❌ Error: Please set JSON_FOLDER and AUDIO_FOLDER to your actual paths!")
        print("Edit the script and modify the following variables:")
        print("  - JSON_FOLDER: folder containing .json files")
        print("  - AUDIO_FOLDER: folder containing .mp3 files")
    else:
        # 验证路径存在
        if not os.path.exists(JSON_FOLDER):
            print(f"❌ Error: JSON_FOLDER does not exist: {JSON_FOLDER}")
        elif not os.path.exists(AUDIO_FOLDER):
            print(f"❌ Error: AUDIO_FOLDER does not exist: {AUDIO_FOLDER}")
        else:
            train(args.num_epochs, data_fraction=args.data_fraction, checkpoint_dir=args.checkpoint_dir, final_model_dir=args.final_model_dir)
