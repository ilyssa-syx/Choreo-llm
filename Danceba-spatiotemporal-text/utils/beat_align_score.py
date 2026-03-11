import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt 

music_root = './data/aistpp_test_full_wav'


def should_skip_ch01(name, base_dir):
    if 'ch01' not in name:
        return False
    alt = name.replace('ch01', 'ch02')
    # 如果存在对应的 ch02 文件，则跳过 ch01 文件
    if os.path.exists(os.path.join(base_dir, alt)):
        return True
    return False

def should_process_file(name, base_dir):
    """检查文件是否应该参与计算"""
    # 文件名第一个字母必须是 'g'
    if name.startswith('g'):
        return True
    return False
    # 不跳过 ch01 文件（如果 should_skip_ch01 返回 True 则跳过）
    if should_skip_ch01(name, base_dir):
        return False
    return True


def upsample_30_to_60(motion):
    """
    使用线性插值将30fps的动作上采样到60fps
    Args:
        motion: (T, D) 形状的数组
    Returns:
        upsampled motion: (2*T-1, D) 形状的数组
    """
    from scipy.interpolate import interp1d
    T = motion.shape[0]
    D = motion.shape[1] if len(motion.shape) > 1 else 1
    # 原始时间点
    x_old = np.arange(T)
    # 新的时间点（在原始点之间插入新点）
    x_new = np.linspace(0, T-1, 2*T-1)
    # 对每个维度进行插值
    f = interp1d(x_old, motion, axis=0, kind='linear')
    return f(x_new)


def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')


        # print(len(beats))
        return beat_axis


def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(root, upsample=False):

    # gt_list = []
    ba_scores = []

    for pkl in os.listdir(root):
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        
        # 检查文件是否应该处理（必须以'g'开头，且不是应跳过的ch01文件）
        if not should_process_file(pkl, root):
            print(f"Skipping {pkl} (doesn't start with 'g' or is ch01 with ch02)")
            continue
        
        print(f"Processing {pkl}")
        try:
            joint3d = np.load(os.path.join(root, pkl), allow_pickle=True)['full_pose'][:, :]
        except:
            joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:1200,:]
        # 如果需要上采样，将30fps插值到60fps
        if upsample:
            joint3d = upsample_30_to_60(joint3d)
            print(f"Upsampled {pkl} from 30fps to 60fps")

        dance_beats, length = calc_db(joint3d, pkl)  
        try:      
            music_beats = get_mb(pkl.split('.')[0][5:] + '.json', length)
        except:
            music_beats = get_mb(pkl.split('.')[0] + '.json', length)
        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':

    # aa = np.random.randn(39, 72)*
    # bb = np.random.randn(39, 72)*0.1
    # print(calc_fid(aa, bb))
    # gt_root = '/mnt/lustre/lisiyao1/dance/bailando/aist_features_zero_start'
    # pred_root = '/mnt/lustressd/lisiyao1/dance_experiements/experiments/sep_vqvae_root_global_vel_wav_acc_batch8/vis/pkl/ep000500'
    # pred_root = ''
    pred_root = '/network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text/experiments/cc_motion_gpt_text_fix_choreo_aist_maintable/choreo_complete/maintable/ep000135'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_gpt_ds8_lbin512_c512_di3full/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/eval/pkl/ep000020'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav_bsz_16_layer6/eval/pkl/ep000040'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/sep_vqvae_root_data_l1_d8_local_c512_di3_global_vel_full_beta0.9_1e-4_wav_beta0.5/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_666_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000080'
    # print('Calculating and saving features')
    
    # 开关：设置 upsample=True 来启用30fps到60fps的线性插值上采样
    print(calc_ba_score(pred_root, upsample=False))
    # calc_and_save_feats(gt_root)

    # print('Calculating metrics')
    # print(gt_root)
    # print(pred_root)
    # print(quantized_metrics(pred_root, gt_root))
