import json
import os
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
from utils.motion_beat_detection import extract_motion_beats
from utils.get_keypoints import get_keypoints_from_smpl

def process_single_file(filename, segment_dir, pkl_dir, output_dir, skip_existing=True):
    """处理单个文件的函数"""
    try:
        bsnm = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, bsnm + '.json')
        
        # 检查输出文件是否已存在
        if skip_existing and os.path.exists(output_path):
            print(f"[SKIP] {bsnm} - 文件已存在")
            return f"skipped: {bsnm}"
        
        print(f"[Processing] {bsnm}")
        
        # 读取segment文件
        segment_path = os.path.join(segment_dir, filename)
        with open(segment_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        # 读取pkl文件
        pkl_path = os.path.join(pkl_dir, bsnm + '.pkl')
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        
        # 提取关键帧
        keypoints_segment = get_keypoints_from_smpl(pkl_data)
        
        keyframes = []
        for segment in segments:
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            
            keyframe = extract_motion_beats(
                keypoints_segment[start_frame:end_frame, :, :], 
                starting_point=start_frame
            ).astype(int).tolist()
            
            keyframe.append(start_frame)
            keyframe.append(end_frame - 1)
            keyframe = sorted(set(keyframe))
            
            segment['key_frame'] = keyframe
            keyframes.append(segment)
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(keyframes, f, indent=4, ensure_ascii=False)
        
        print(f"[DONE] {bsnm}")
        return f"success: {bsnm}"
        
    except Exception as e:
        print(f"[ERROR] {filename}: {str(e)}")
        return f"error: {filename} - {str(e)}"


def main():
    # 配置参数
    segment_dir = '../i3d_18_segmentation/test'
    pkl_dir = '/network_space/server126/shared/sunyx/models/Edge-spatiotemporal-text/data/test/motions_sliced'
    output_dir = 'test'
    skip_existing = True  # 是否跳过已存在的文件
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有待处理文件
    files = [f for f in os.listdir(segment_dir) if f.endswith('.json')]
    
    print(f"找到 {len(files)} 个文件待处理")
    print(f"使用 {cpu_count()} 个CPU核心进行并行处理")
    print("-" * 50)
    
    # 创建部分函数，固定部分参数
    process_func = partial(
        process_single_file,
        segment_dir=segment_dir,
        pkl_dir=pkl_dir,
        output_dir=output_dir,
        skip_existing=skip_existing
    )
    
    # 使用进程池并行处理
    # 可以根据实际情况调整进程数，建议使用 cpu_count() 或 cpu_count() - 1
    num_processes = max(1, cpu_count() - 1)
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_func, files)
    
    # 统计结果
    print("\n" + "=" * 50)
    print("处理完成！")
    success_count = sum(1 for r in results if r.startswith("success"))
    skipped_count = sum(1 for r in results if r.startswith("skipped"))
    error_count = sum(1 for r in results if r.startswith("error"))
    
    print(f"成功: {success_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {error_count}")
    
    # 显示错误详情
    if error_count > 0:
        print("\n错误详情:")
        for r in results:
            if r.startswith("error"):
                print(f"  - {r}")


if __name__ == "__main__":
    main()