import json
from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
import clip
import re
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import functools
import argparse

def encode_text_batch(model, texts, device, batch_size=256, zero_features=False):
    """
    批量使用CLIP模型对文本进行编码
    Args:
        texts: 文本列表
        batch_size: 批处理大小
        zero_features: 若为True，不提取特征，直接返回全0
    Returns:
        embeddings: [N, 512] numpy array, N个文本的embedding
    """
    if not texts:
        return np.array([], dtype=np.float32)

    if zero_features:
        return np.zeros((len(texts), 512), dtype=np.float32)
    
    all_features = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 处理空文本
        batch_tokens = []
        empty_indices = []
        for idx, text in enumerate(batch_texts):
            if not text or text.strip() == "":
                empty_indices.append(idx)
                batch_tokens.append("")  # 占位
            else:
                batch_tokens.append(text)
        
        # 对批量文本进行tokenize
        tokens = clip.tokenize(batch_tokens, truncate=True).to(device)
        
        with torch.no_grad():
            # 批量编码
            text_features = model.encode_text(tokens)  # [batch_size, 512]
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        batch_features = text_features.cpu().numpy().astype(np.float32)
        
        # 将空文本对应的特征设为0
        for idx in empty_indices:
            batch_features[idx] = np.zeros(512, dtype=np.float32)
        
        all_features.append(batch_features)
    
    return np.vstack(all_features) if all_features else np.array([], dtype=np.float32)

def process_json_simple(json_path, model, device, output_dir, zero_features=False, overwrite=False):
    """
    处理单个JSON文件，提取CLIP特征（简化版）
    支持处理空文件
    Args:
        overwrite: 是否覆盖已存在的输出文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件名（不带后缀）用于命名输出文件
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    
    # 检查输出文件是否已存在
    npy_path = os.path.join(output_dir, f"{base_name}_features.npz")
    json_out_path = os.path.join(output_dir, f"{base_name}_meta.json")
    
    if not overwrite and os.path.exists(npy_path) and os.path.exists(json_out_path):
        print(f"⚠ Output already exists for {base_name}, skipping")
        return []
    
    # 读取JSON文件
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    # 检查是否为空文件
    if not data or len(data) == 0:
        print(f"⚠ Empty JSON file: {base_name}, creating empty outputs")
        # 创建空的输出文件
        np.savez_compressed(npy_path)  # 空的npz文件
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f"✓ Created empty outputs for {base_name}")
        return []
    
    # 准备一个大字典来存所有的特征
    save_dict = {}
    metadata_list = []  # 存轻量级信息
    
    # 第一遍：收集所有文本和元数据
    all_texts = []
    text_keys = []  # 保存对应的key
    part_order = ["whole", "upper_half_body", "lower_half_body", "torso"]
    idx = 0
    
    
        # 自动检测JSON文件的组织形式
    if 'segments' in data:
            # 格式1: data['segments'] -> sent['sentence'] -> seg
            segments_list = []
            for sent in data['segments']:
                if 'sentence' in sent:
                    segments_list.extend(sent['sentence'])
    elif 'merged_segments' in data:
            # 格式2: data['merged_segments'] -> seg
            segments_list = data['merged_segments']
    else:
            raise ValueError(f"Unknown JSON structure. Expected 'segments' or 'merged_segments' key in data.")
        
        # 统一处理所有segments
    for seg in segments_list:
            for part in part_order:
                part_text = seg['modifier'].get(part, '')
                all_texts.append(part_text)
                if part == 'upper_half_body' or part == 'lower_half_body':
                    tempkey = part[:5] + '_body'
                else:
                    tempkey = part
                text_keys.append(f"{idx}_{tempkey}_feat")
            
            # 记录元数据（轻量级）
            metadata_list.append({
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'original_index': idx
            })
            
            idx += 1
    
    
    # 第二遍：批量编码所有文本
    print(f"Encoding {len(all_texts)} texts for {base_name}...")
    try:
        all_embeddings = encode_text_batch(model, all_texts, device, zero_features=zero_features)
    except Exception as e:
        print(f"❌ Error encoding texts for {json_path}: {e}")
        return []
    
    # 第三遍：将编码结果存入字典
    for key, emb in zip(text_keys, all_embeddings):
        save_dict[key] = emb
    
    # 1. 保存特征数据（重资产）-> .npz
    try:
        np.savez_compressed(npy_path, **save_dict)
        print(f"✓ Saved features to {npy_path}")
    except Exception as e:
        print(f"❌ Error saving features for {json_path}: {e}")
        return []
    
    # 2. 保存元数据（轻资产）-> .json
    try:
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved metadata to {json_out_path}")
    except Exception as e:
        print(f"❌ Error saving metadata for {json_path}: {e}")
        return []
    
    return metadata_list


def process_single_file(args):
    """
    处理单个JSON文件的包装函数，用于多进程
    添加了异常处理以防止单个文件错误导致进程崩溃
    """
    json_file, input_path, output_path, device_id, zero_features, overwrite = args
    
    try:
        # 若只需要全0特征，避免加载CLIP模型/占用GPU显存
        if zero_features:
            model = None
            device = "cpu"
        else:
            # 每个进程使用独立的GPU
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            # 每个进程加载自己的模型
            model, _ = clip.load("ViT-B/32", device=device)
        
        # 构建输出路径
        relative_path = json_file.relative_to(input_path)
        output_subdir = output_path / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # 处理文件
        process_json_simple(str(json_file), model, device, str(output_subdir), zero_features=zero_features, overwrite=overwrite)
        return f"✓ {str(json_file)}"
    
    except Exception as e:
        error_msg = f"❌ Error processing {str(json_file)}: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def process_directory(input_dir, output_dir, num_workers=4, zero_features=False, overwrite=False):
    """
    处理整个目录中的所有JSON文件，支持多进程并行
    Args:
        num_workers: 并行进程数，默认4
        overwrite: 是否覆盖已存在的输出文件
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = list(input_path.rglob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Using {num_workers} workers for parallel processing")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")
    
    # 准备参数列表，分配GPU
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args_list = [(json_file, input_path, output_path, i % num_gpus, zero_features, overwrite) 
                 for i, json_file in enumerate(json_files)]
    
    # 使用多进程处理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, args_list), 
                           total=len(json_files), 
                           desc="Processing files"))
    
    # 统计结果
    success_count = sum(1 for r in results if r.startswith("✓"))
    error_count = sum(1 for r in results if r.startswith("❌"))
    
    print(f"\n{'='*60}")
    print(f"Processing completed:")
    print(f"  Total files: {len(json_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")


def main():
    # 设置多进程启动方法为spawn以支持CUDA
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过就忽略
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Extract CLIP text features from JSON files')
    parser.add_argument('--input_dir', type=str, default='./choreo_original/text',
                        help='Input directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='./choreo_original/text_features',
                        help='Output directory for features and metadata')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto based on GPU count)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files (default: skip existing files)')
    parser.add_argument('--zero_features', action='store_true',
                        help='Generate zero features instead of using CLIP model')
    
    args = parser.parse_args()
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    zero_features = args.zero_features
    overwrite = args.overwrite
    
    # 设置并行worker数量（可以根据GPU数量调整）
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = max(1, num_gpus * 2) if num_gpus > 0 else 4
    
    process_directory(input_dir, output_dir, num_workers=num_workers, 
                     zero_features=zero_features, overwrite=overwrite)


if __name__ == "__main__":
    main()