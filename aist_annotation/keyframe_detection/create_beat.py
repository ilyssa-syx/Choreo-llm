import json
import os
import pickle
from utils.motion_beat_detection import extract_motion_beats
from utils.get_keypoints import get_keypoints_from_smpl

# 1. 打开并读取文件
segment_dir = '../i3d_18_segmentation/test'
pkl_dir = '/network_space/storage43/sunyixuan/models/EDGE/data/train/motions_sliced'
output_dir = 'test'

for filename in os.listdir(segment_dir):
    segment_path = os.path.join(segment_dir, filename)
    with open(segment_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    bsnm = os.path.splitext(os.path.basename(segment_path))[0]
    print(bsnm)
    pkl_path = os.path.join(pkl_dir, bsnm + '.pkl')
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    
    keyframes = []
    for segment in segments:
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        keypoints_segment = get_keypoints_from_smpl(pkl_data)
        keyframe = extract_motion_beats(keypoints_segment[start_frame:end_frame, :, :], starting_point=start_frame).astype(int).tolist()
        keyframe.append(start_frame)
        keyframe.append(end_frame-1)
        keyframe = sorted(set(keyframe))

        print(keyframe)
        segment['key_frame'] = keyframe
        keyframes.append(segment)
    
    output_path = os.path.join(output_dir, bsnm + '.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keyframes, f, indent=4, ensure_ascii=False)

    