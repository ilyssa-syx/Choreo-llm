# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor
# from aistplusplus_api.aist_plusplus.loader import AISTDataset
from smplx import SMPL
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_audio_dir', type=str, default='/network_space/server127/shared/sunyx/FineDance/data/finedance/music_wav/test')
parser.add_argument('--input_motion_dir', type=str, default='/network_space/server127/shared/sunyx/FineDance/data/finedance/motion/test')
parser.add_argument('--smpl_path', type=str, default='/network_space/server127_2/shared/sunyx3/tools/smpl/SMPL_MALE.pkl')

parser.add_argument('--train_dir', type=str, default='data/finedance_train_wav')
parser.add_argument('--test_dir', type=str, default='data/finedance_test_full_wav')

parser.add_argument('--sampling_rate', type=int, default=15360*2)
args = parser.parse_args()

extractor = FeatureExtractor()

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

def ax_from_6v(q):
    # q: (N,6) torch -> returns (N,3) axis-angle
    mat = rotation_6d_to_matrix(q)       # (N,3,3)
    ax = matrix_to_axis_angle(mat)       # (N,3)
    return ax


if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)

def load_data(audio_name):
    data = np.load(os.path.join(args.input_motion_dir, audio_name + '.npy'))  # expect shape (T, C) where C >= 3 and rest are 6D rotations
    T = data.shape[0]

    rot6d = torch.from_numpy(data[:, 3:]).float()            # (T, M)
    rot6d = rot6d.reshape(-1, 6)                             # (T * (M/6), 6)
    axis = ax_from_6v(rot6d).view(T, -1).detach().cpu().numpy()  # (T, M/2 * 3)

    modata = np.concatenate([data[:, :3], axis], axis=1)     # (T, 3 + axis_len)

    smpl_trans = modata[:, :3]
    smpl_poses = modata[:, 3:69]    # (T, 66)

    left_hand = modata[:, 69:114].reshape(T, 15, 3)
    right_hand = modata[:, 114:159].reshape(T, 15, 3)
    lhand_avg = left_hand.mean(axis=1)   # (T,3)
    rhand_avg = right_hand.mean(axis=1)  # (T,3)

    smpl_poses = np.concatenate([smpl_poses, lhand_avg, rhand_avg], axis=1)  # (T,72)

    return smpl_poses.astype(np.float32), 1.0, smpl_trans.astype(np.float32)
    

def make_music_dance_set():
    print('---------- Extract features from raw audio ----------')
    # print(annotation_dir)
    

    musics = []
    dances = []
    fnames = []
    train = []
    test = []

    # music_dance_keys = []

    # onset_beats = []
    
    # dance_fnames = sorted(os.listdir(dance_dir))
    # audio_fnames = audio_fnames[:20]  # for debug
    # print(f'audio_fnames: {audio_fnames}')

    
    ii = 0
    
    for audio_file in os.listdir(args.input_audio_dir):
        
        loader = None
        try:
            loader = essentia.standard.MonoLoader(filename=os.path.join(args.input_audio_dir, audio_file), sampleRate=args.sampling_rate)
            print('successfully loaded audio file')
        except RuntimeError:
            continue

        fnames.append(audio_file.split('.')[0])
        print(audio_file.split('.')[0])
        
        ### load audio features ###

    
        audio = loader()
        audio = np.array(audio).T

        feature =  extract_acoustic_feature(audio, args.sampling_rate)
        musics.append(feature.tolist())

        ### load pose sequence ###
       # for seq_name in tqdm(seq_names):
        
        smpl_poses, smpl_scaling, smpl_trans = load_data(audio_file.split('.')[0])
        smpl = None
        smpl = SMPL(model_path=args.smpl_path, gender='MALE', batch_size=1)
        keypoints3d = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
            ).joints.detach().numpy()[:, 0:24, :]
        
        # 从30fps插值到60fps
        nframes = keypoints3d.shape[0]
        original_indices = np.arange(nframes)
        new_nframes = nframes * 2 - 1  # 插值后的帧数
        new_indices = np.linspace(0, nframes - 1, new_nframes)
        
        # 向量化线性插值：将(nframes, 24, 3)展平为(nframes, 72)后一次性插值
        keypoints3d_flat = keypoints3d.reshape(nframes, -1)  # (nframes, 72)
        keypoints3d_interpolated = np.array([
            np.interp(new_indices, original_indices, keypoints3d_flat[:, i]) 
            for i in range(72)
        ]).T  # (new_nframes, 72)
        
        keypoints3d = keypoints3d_interpolated.reshape(new_nframes, 24, 3)
        nframes = keypoints3d.shape[0]
        dances.append(keypoints3d.reshape(nframes, -1).tolist())
        print(np.shape(dances[-1]))  # (nframes, 72)

    # return None, None, None
    return musics, dances, fnames


def extract_acoustic_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr)
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr, octave=7 if sr==15360*2 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, sr=sr)
    tempogram = extractor.get_tempogram(onset_env, sr=sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr=sr)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc, # 20
        mfcc_delta, # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
        tempogram
    ], axis=0)

            # mfcc, #20
            # mfcc_delta, #20

            # chroma_cqt, #12
            # onset_env, # 1
            # onset_beat, #1

    feature = feature.transpose(1, 0)
    print(f'acoustic feature -> {feature.shape}')

    return feature

def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = min(len(musics[i]), len(dances[i]))
        print(f'music -> {np.array(musics[i]).shape}, ' +
              f'dance -> {np.array(dances[i]).shape}, ' +
              f'min_seq_len -> {min_seq_len}')

        new_musics.append([musics[i][j] for j in range(min_seq_len)])
        new_dances.append([dances[i][j] for j in range(min_seq_len)])

    return new_musics, new_dances



def save(args, musics, dances, fnames):
    print('---------- Save to text file ----------')
    # fnames = sorted(os.listdir(os.path.join(args.input_dance_dir,inner_dir)))
    # # fnames = fnames[:20]  # for debug
    # assert len(fnames)*2 == len(musics) == len(dances), 'alignment'

    fnames = sorted(fnames)
    for idx in fnames:
        with open(os.path.join(args.test_dir, f'{idx}.json'), 'w') as f:
            sample_dict = {
                'id': idx,
                'music_array': musics[fnames.index(idx)],
                'dance_array': dances[fnames.index(idx)]
            }
            # print(sample_dict)
            json.dump(sample_dict, f)

    

if __name__ == '__main__':
    musics, dances, fnames = make_music_dance_set() 
    print(fnames)
    musics, dances = align(musics, dances)
    save(args, musics, dances, fnames)


