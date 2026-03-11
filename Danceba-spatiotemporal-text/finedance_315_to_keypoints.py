#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理动作数据：6D旋转 -> 轴角表示 -> SMPL前向传播 -> keypoints3d
输出格式：[T, 72] 的 .npy 文件
"""

import os
import argparse
import numpy as np
import torch
from smplx import SMPL
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
from tqdm import tqdm


def ax_from_6v(q):
    """
    将6D旋转表示转换为轴角表示
    Args:
        q: (N, 6) torch tensor
    Returns:
        (N, 3) 轴角表示
    """
    mat = rotation_6d_to_matrix(q)       # (N, 3, 3)
    ax = matrix_to_axis_angle(mat)       # (N, 3)
    return ax


def load_motion_data(motion_file):
    """
    加载动作数据并转换为轴角表示
    Args:
        motion_file: 动作文件路径
    Returns:
        smpl_poses: (T, 72) SMPL姿势参数（包含手部平均位置）
        smpl_trans: (T, 3) SMPL平移参数
        smpl_scaling: 缩放因子
    """
    # 加载数据，期望形状为 (T, C)，其中前3列是平移，其余是6D旋转
    data = np.load(motion_file)
    T = data.shape[0]
    
    # 提取6D旋转部分并转换为轴角
    rot6d = torch.from_numpy(data[:, 3:]).float()            # (T, M)
    rot6d = rot6d.reshape(-1, 6)                             # (T * (M/6), 6)
    axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()  # (T, M/2 * 3)
    
    # 合并平移和轴角表示
    modata = np.concatenate([data[:, :3], axis], axis=1)     # (T, 3 + axis_len)
    
    smpl_trans = modata[:, :3]
    smpl_poses = modata[:, 3:69]    # (T, 66) - 身体姿势
    
    # 处理手部数据：计算左右手的平均位置
    left_hand = modata[:, 69:114].reshape(T, 15, 3)
    right_hand = modata[:, 114:159].reshape(T, 15, 3)
    lhand_avg = left_hand.mean(axis=1)   # (T, 3)
    rhand_avg = right_hand.mean(axis=1)  # (T, 3)
    
    # 拼接为完整的SMPL姿势参数
    smpl_poses = np.concatenate([smpl_poses, lhand_avg, rhand_avg], axis=1)  # (T, 72)
    
    return smpl_poses.astype(np.float32), smpl_trans.astype(np.float32), 1.0


def motion_to_keypoints3d(motion_file, smpl_model, smpl_scaling=1.0):
    """
    将动作数据转换为3D关键点
    Args:
        motion_file: 输入动作文件路径
        smpl_model: SMPL模型
        smpl_scaling: SMPL缩放因子
    Returns:
        keypoints3d: (T, 72) 形状的3D关键点数组
    """
    # 加载并转换动作数据
    smpl_poses, smpl_trans, smpl_scaling = load_motion_data(motion_file)
    
    # SMPL前向传播得到关节位置
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
    ).joints.detach().numpy()[:, 0:24, :]  # (T, 24, 3) - 取前24个关节
    
    # Reshape为 (T, 72)
    T = keypoints3d.shape[0]
    keypoints3d = keypoints3d.reshape(T, -1)  # (T, 72)
    
    return keypoints3d


def process_motion_folder(input_dir, output_dir, smpl_path, gender='MALE'):
    """
    批量处理动作文件夹
    Args:
        input_dir: 输入动作文件夹路径
        output_dir: 输出文件夹路径
        smpl_path: SMPL模型路径
        gender: SMPL性别参数
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化SMPL模型
    print(f'加载SMPL模型: {smpl_path}')
    smpl = SMPL(model_path=smpl_path, gender=gender, batch_size=1)
    
    # 获取所有.npy文件
    motion_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    print(f'找到 {len(motion_files)} 个动作文件')
    
    # 处理每个文件
    for motion_file in tqdm(motion_files, desc='处理动作文件'):
        input_path = os.path.join(input_dir, motion_file)
        output_path = os.path.join(output_dir, motion_file)
        
        try:
            # 转换为keypoints3d
            keypoints3d = motion_to_keypoints3d(input_path, smpl)
            
            # 保存结果
            np.save(output_path, keypoints3d)
            
            print(f'✓ {motion_file}: {keypoints3d.shape}')
            
        except Exception as e:
            print(f'✗ {motion_file}: 处理失败 - {str(e)}')
            continue
    
    print(f'\n处理完成！输出保存至: {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='处理动作数据：6D旋转 -> SMPL -> keypoints3d')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入动作文件夹路径（包含.npy文件）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出文件夹路径')
    parser.add_argument('--smpl_path', type=str, 
                        default='/network_space/server127_2/shared/sunyx3/tools/smpl/SMPL_MALE.pkl',
                        help='SMPL模型路径')
    parser.add_argument('--gender', type=str, default='MALE',
                        choices=['MALE', 'FEMALE', 'NEUTRAL'],
                        help='SMPL性别参数')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input_dir):
        print(f'错误：输入文件夹不存在: {args.input_dir}')
        return
    
    if not os.path.exists(args.smpl_path):
        print(f'错误：SMPL模型文件不存在: {args.smpl_path}')
        return
    
    # 处理动作文件
    process_motion_folder(args.input_dir, args.output_dir, args.smpl_path, args.gender)


if __name__ == '__main__':
    main()
