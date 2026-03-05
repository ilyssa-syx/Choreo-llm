import cv2
import numpy as np

def read_and_slice_video(video_path, stride, length, num_slices, target_size=256, crop_size=224):
    """
    读取视频并按照指定参数切片
    
    Args:
        video_path: 视频文件路径
        stride: 切片步长(秒)
        length: 每个切片长度(秒)
        num_slices: 最大切片数量
        target_size: resize的目标尺寸
        crop_size: 中心裁剪的尺寸
    
    Returns:
        list: 包含多个视频片段的列表，每个片段是 numpy array (T, C, H, W)
    """
    video_path = video_path.replace("_cAll_", "_c01_")

    # 读取整个视频(已经抽帧到30fps)
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 只保留偶数帧，从60fps降到30fps
        if frame_idx % 2 == 0:
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            start = (target_size - crop_size) // 2
            frame = frame[start:start+crop_size, start:start+crop_size]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = frame.transpose(2, 0, 1)  # (C, H, W)
            frames.append(frame)
        
        frame_idx += 1
        
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")
    
    # 转换为numpy数组 (T, C, H, W)
    all_frames = np.stack(frames, axis=0)
    
    print(f'Video path: {video_path}')
    print(f'Total frames after downsampling to 30fps: {all_frames.shape[0]}')
    
    # 开始切片
    video_slices = []
    window = int(length * 30)  # 30fps
    stride_step = int(stride * 30)  # 30fps
    start_idx = 0
    slice_count = 0
    
    while start_idx <= len(all_frames) - window and slice_count < num_slices:
        video_slice = all_frames[start_idx : start_idx + window]
        video_slices.append(video_slice)
        print(f'Slice {slice_count}: frames {start_idx} to {start_idx + window}, shape: {video_slice.shape}')
        start_idx += stride_step
        slice_count += 1
    
    print(f'Total slices created: {len(video_slices)}')
    return video_slices