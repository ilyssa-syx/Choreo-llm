# 1920*1080
from utils.pytorch_i3d import InceptionI3d
import cv2
import os
import numpy as np
import torch
from pathlib import Path

def read_video_frames(video_path, target_size=256, crop_size=224):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        start = (target_size - crop_size) // 2
        frame = frame[start:start+crop_size, start:start+crop_size]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (frame.astype(np.float32) / 255.0) * 2 - 1
        frame = frame.transpose(2, 0, 1)
        frames.append(frame)
        frame_idx += 1
        
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames read from video.")
    arr = np.stack([f for f in frames], axis=0)  # (T, C, H, W)
    arr = arr[::2, :, :, :]
    """
    total_frames = arr.shape[0]
    valid_len = (total_frames // 8) * 8
    arr = arr[:valid_len]
    """
    print('path:', video_path)
    print('len:', arr.shape[0])
    return arr  # dtype float32, values in [0,1]

def prepare_input_tensor(frames_TCHW):
    t, c, h, w = frames_TCHW.shape
    # make shape (C, T, H, W)
    arr = frames_TCHW.transpose(1, 0, 2, 3).astype(np.float32)
    tensor = torch.from_numpy(arr)  # (C, T, H, W)
    tensor = tensor.unsqueeze(0)    # (1, C, T, H, W)
    return tensor

def extract_features(i3d, frames_TCHW):
    """
    Extract features with specified density multiplier.
    
    Args:
        i3d: I3D model
        frames_TCHW: numpy array of shape (T, C, H, W)
        density_multiplier: 1 for base (18 frames), 2 for 36 frames, 4 for 72, etc.
    
    Returns:
        features: numpy array of shape (N, 1024) where N = base_frames * density_multiplier
    """
    # Original behavior: just extract features normally
    input_tensor = prepare_input_tensor(frames_TCHW).cuda()
    feats = i3d.extract_features(input_tensor)
    # feats: (1, 1024, T'', 1, 1) — avg_pool keeps spatial dims as 1x1
    # squeeze batch + spatial dims -> (1024, T''), then transpose -> (T'', 1024)
    feats = feats.squeeze(0).squeeze(2).squeeze(2)  # (1024, T'')
    feats = feats.permute(1, 0)                      # (T'', 1024)
    return feats.data.cpu().numpy()

def process_video_dir(video_dir, save_dir, i3d, density_multiplier=1):
    video_dir = Path(video_dir)
    save_dir = Path(save_dir)
    
    video_files = list(video_dir.rglob('*.mp4')) + list(video_dir.rglob('*.avi'))
    
    for video_path in video_files:
        try:
            rel_path = video_path.relative_to(video_dir)
            output_dir = save_dir / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            bsnm = video_path.stem
            # Add density suffix to filename
            
            save_path = output_dir / f'{bsnm}.npy'
            """
            if save_path.exists():
                print(f'Skip (exists): {rel_path}')
                continue
            """
                
            print(f'Processing: {rel_path} (density x{density_multiplier})')
            frames_TCHW = read_video_frames(str(video_path))
            
            feats = extract_features(i3d, frames_TCHW, density_multiplier)
            
            np.save(save_path, feats)
            print(f'Saved: {save_path} {feats.shape}')
            
        except Exception as e:
            print(f'Error processing {video_path}: {e}')



def initialize_i3d():
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load('utils/rgb_charades.pt'))
    i3d.cuda()
    i3d.eval()
    return i3d