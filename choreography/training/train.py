import os
import json
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import shutil
import glob
import argparse

# 导入自定义的Dataset类
from dataset import DanceSongDataset, DanceSegmentDataset, DanceOnestageDataset

# ============= 配置参数 =============
# 请修改以下路径为实际路径
JSON_FOLDER = "/network_space/server126/shared/sunyx/datasets/aist/text_description/song/train"  # 训练集JSON文件所在文件夹
JSON_FOLDER_TEST = "/network_space/server126/shared/sunyx/datasets/aist/text_description/song/test"  # 测试集JSON文件所在文件夹
AUDIO_FOLDER = "/network_space/server127_2/shared/caixhdata/aist_w/"  # MP3音频文件所在文件夹

MODEL_ID = "/network_space/server127_2/shared/sunyx3/huggingface/hub/models--nvidia--music-flamingo-hf/snapshots/e29cfe92e682616f8f8014c60b2c5d17a37d4e33"
BATCH_SIZE = 1  # 音频编码器很吃显存，单卡 batch=1 是防 OOM 的底线
# NUM_EPOCHS 移除，改为命令行参数

# ============= Ablation 配置 =============
ABLATION = "FULL"  # "FULL" 或 "NOCOT" (移除思考过程)

# ============= Stage 配置 =============
STAGE = "TWOSTAGE"  # "TWOSTAGE" (Song + Segment) 或 "ONESTAGE" (Onestage only)

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


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_dir):
    """保存完整的checkpoint（模型+训练状态）"""
    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")

    """
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
    """
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


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
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


def evaluate_on_test_set(model, test_dataloader, processor, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader, desc="Test evaluation"):
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
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    
    return avg_loss, perplexity


def train(num_epochs, data_fraction=None, checkpoint_dir=None, final_model_dir=None):
    """训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Training Setup")
    print(f"{'='*60}")
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
        print("✓ 4-bit quantization enabled")
    
    # 2. 加载处理器
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 确保 Tokenizer 有 pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        print("✓ Set pad_token to eos_token")
    
    # 3. 加载模型
    try:
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            # 如果支持 flash attention，使用它来加速和节省显存
            attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Failed to load with flash_attention_2, falling back to eager: {e}")
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    # 设置 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # 关闭cache，启用输入梯度
    model.config.use_cache = False
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    print("✓ Settings applied (use_cache=False, input_require_grads=True)")
    
    # 4. 启用梯度检查点 (极其重要！节省大量显存)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("✓ Gradient checkpointing enabled (use_reentrant=False)")
    
    # 5. LoRA 配置和应用
    if USE_LORA:
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
        model.print_trainable_parameters()
        print("✓ LoRA applied")
    
    model.train()
    
    # 创建训练数据集
    print("\nCreating training datasets...")
    print(f"Stage: {STAGE}")
    
    if STAGE == "TWOSTAGE":
        # 两阶段训练：先生成high-level choreography，再生成详细动作
        song_dataset = DanceSongDataset(JSON_FOLDER, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Song-level dataset size: {len(song_dataset)}")
        
        segment_dataset = DanceSegmentDataset(JSON_FOLDER, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Segment-level dataset size: {len(segment_dataset)}")
        
        # 合并两个数据集
        dataset = ConcatDataset([song_dataset, segment_dataset])
    elif STAGE == "ONESTAGE":
        # 一阶段训练：直接从音乐生成详细动作
        onestage_dataset = DanceOnestageDataset(JSON_FOLDER, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Onestage dataset size: {len(onestage_dataset)}")
        dataset = onestage_dataset
    else:
        raise ValueError(f"Invalid STAGE: {STAGE}. Must be 'TWOSTAGE' or 'ONESTAGE'")
    
    # 创建测试数据集
    print("\nCreating test datasets...")
    if STAGE == "TWOSTAGE":
        test_song_dataset = DanceSongDataset(JSON_FOLDER_TEST, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Test song-level dataset size: {len(test_song_dataset)}")
        
        test_segment_dataset = DanceSegmentDataset(JSON_FOLDER_TEST, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Test segment-level dataset size: {len(test_segment_dataset)}")
        
        test_dataset = ConcatDataset([test_song_dataset, test_segment_dataset])
    elif STAGE == "ONESTAGE":
        test_onestage_dataset = DanceOnestageDataset(JSON_FOLDER_TEST, AUDIO_FOLDER, processor, ablation=ABLATION)
        print(f"Test onestage dataset size: {len(test_onestage_dataset)}")
        test_dataset = test_onestage_dataset
    print(f"Total test dataset size: {len(test_dataset)}")
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
        print(
            f"Using data fraction: {fraction} ({subset_size}/{total_size}), "
            f"strategy={DATA_SAMPLING_STRATEGY}"
        )
    print(f"Total dataset size: {len(dataset)}")
    
    # DEBUG: 输出数据集样本信息
    if DEBUG_MODE:
        print("\n" + "="*80)
        print("DEBUG MODE: Inspecting dataset samples")
        print(f"Stage: {STAGE}")
        print(f"Showing level: {DEBUG_SHOW_LEVEL}")
        print("="*80)
        
        debug_output_lines = []
        debug_output_lines.append("\n" + "="*80 + "\n")
        debug_output_lines.append("DEBUG MODE: Dataset Sample Inspection\n")
        debug_output_lines.append("="*80 + "\n")
        debug_output_lines.append(f"Generated at: {__import__('datetime').datetime.now()}\n")
        debug_output_lines.append(f"Stage: {STAGE}\n")
        
        if STAGE == "TWOSTAGE":
            debug_output_lines.append(f"Song-level dataset size: {len(song_dataset)}\n")
            debug_output_lines.append(f"Segment-level dataset size: {len(segment_dataset)}\n")
        elif STAGE == "ONESTAGE":
            debug_output_lines.append(f"Onestage dataset size: {len(onestage_dataset)}\n")
        debug_output_lines.append(f"Total dataset size: {len(dataset)}\n\n")
        
        # Segment 样本（优先展示）- 仅在 TWOSTAGE 模式下
        if STAGE == "TWOSTAGE" and DEBUG_SHOW_LEVEL in ("both", "segment"):
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
        
        # Song 样本 - 仅在 TWOSTAGE 模式下
        if STAGE == "TWOSTAGE" and DEBUG_SHOW_LEVEL in ("both", "song"):
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
        
        # Onestage 样本 - 仅在 ONESTAGE 模式下
        if STAGE == "ONESTAGE":
            debug_output_lines.append("\n" + "="*80 + "\n")
            debug_output_lines.append("ONESTAGE DATASET SAMPLES\n")
            debug_output_lines.append("="*80 + "\n")
            for i in range(min(DEBUG_NUM_SAMPLES, len(onestage_dataset))):
                try:
                    conversation, sample_name, dataset_type, debug_info = onestage_dataset[i]
                    print(f"\n{'='*80}")
                    print(f"Onestage Sample {i+1}/{min(DEBUG_NUM_SAMPLES, len(onestage_dataset))}")
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
                    print("助手响应 (Assistant Response):")
                    print(f"{'-'*80}")
                    print(debug_info['assistant_response'][:500] + "..." if len(debug_info['assistant_response']) > 500 else debug_info['assistant_response'])
                    
                    debug_output_lines.append(f"\n{'='*80}\n")
                    debug_output_lines.append(f"Onestage Sample {i+1}\n")
                    debug_output_lines.append(f"JSON文件: {debug_info['json_path']}\n")
                    debug_output_lines.append(f"音频文件: {debug_info['audio_path']}\n")
                    debug_output_lines.append(f"舞蹈风格: {debug_info['genre']}\n")
                    debug_output_lines.append(f"片段数量: {debug_info['num_segments']}\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("用户提示词 (User Prompt):\n")
                    debug_output_lines.append(debug_info['user_prompt'] + "\n")
                    debug_output_lines.append(f"\n{'-'*80}\n")
                    debug_output_lines.append("助手响应 (Assistant Response):\n")
                    debug_output_lines.append(debug_info['assistant_response'] + "\n\n")
                except Exception as e:
                    error_msg = f"\n❌ Error loading onestage sample {i}: {e}\n"
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
            return
    
    # 创建训练集dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建测试集dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 测试集不需要shuffle
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # 7. 使用 Paged AdamW 优化器 (显存不足时自动转移到内存)
    optimizer = bnb.optim.PagedAdamW32bit(model.parameters(), lr=LEARNING_RATE)
    print(f"✓ Optimizer created with lr={LEARNING_RATE}")
    
    # 8. 创建学习率调度器
    scheduler = None
    if USE_SCHEDULER:
        from transformers import get_cosine_schedule_with_warmup
        # 计算总训练步数
        num_training_steps = (len(dataset) // GRADIENT_ACCUMULATION_STEPS) * num_epochs
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"✓ Cosine scheduler created (warmup steps: {num_warmup_steps}/{num_training_steps})")
    
    # 创建检查点目录
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR
    final_dir = final_model_dir if final_model_dir is not None else FINAL_MODEL_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ⚡ 检查并加载最新checkpoint
    start_epoch = 0
    global_step = 0
    
    if RESUME_FROM_CHECKPOINT:
        latest_checkpoint = find_latest_checkpoint(ckpt_dir)
        if latest_checkpoint:
            start_epoch, global_step = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
        else:
            print("\n✓ No checkpoint found, starting training from scratch\n")
    
    # 训练循环
    if start_epoch > 0:
        print(f"\n{'='*60}")
        print(f"Resuming training from epoch {start_epoch}")
        print(f"Remaining epochs: {num_epochs - start_epoch}")
        print(f"{'='*60}\n")
    else:
        print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        total_loss = 0
        optimizer.zero_grad()
        
        # 使用 tqdm 显示进度
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")
        
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
            
            # 更新进度条
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
            }
            pbar.set_postfix(postfix_dict)
        
        # Epoch 结束后的统计
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1} completed")
        print(f"Train Average Loss: {avg_epoch_loss:.4f}")
        print(f"Train Perplexity: {torch.exp(torch.tensor(avg_epoch_loss)).item():.4f}")
        print(f"{'='*50}")
        
        # 在测试集上评估
        test_loss, test_perplexity = evaluate_on_test_set(model, test_dataloader, processor, device)
        print(f"\n{'='*50}")
        print(f"Test Evaluation Results")
        print(f"Test Average Loss: {test_loss:.4f}")
        print(f"Test Perplexity: {test_perplexity:.4f}")
        print(f"{'='*50}")
        
        # 每个 epoch 结束后保存检查点
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, ckpt_dir)
        # 保存processor（只需要保存一次，放在第一个epoch）
        if epoch == start_epoch:
            processor.save_pretrained(os.path.join(ckpt_dir, f"epoch_{epoch + 1}"))
            print(f"✓ Processor saved")
    
    # 保存最终模型
    print(f"\n{'='*50}")
    print("Training completed! Saving final model...")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"✓ Final model saved to: {final_dir}")
    print(f"{'='*50}")


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
    
    if not os.path.exists(JSON_FOLDER):
        print(f"❌ Error: JSON_FOLDER does not exist: {JSON_FOLDER}")
    elif not os.path.exists(JSON_FOLDER_TEST):
        print(f"❌ Error: JSON_FOLDER_TEST does not exist: {JSON_FOLDER_TEST}")
    elif not os.path.exists(AUDIO_FOLDER):
        print(f"❌ Error: AUDIO_FOLDER does not exist: {AUDIO_FOLDER}")
    else:
        train(args.num_epochs, data_fraction=args.data_fraction, checkpoint_dir=args.checkpoint_dir, final_model_dir=args.final_model_dir)
