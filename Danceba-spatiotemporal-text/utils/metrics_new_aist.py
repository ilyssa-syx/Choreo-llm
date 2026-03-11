
import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg

# kinetic, manual
import os

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

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def quantized_metrics(predicted_pkl_root, gt_pkl_root):


    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []


    # for pkl in os.listdir(predicted_pkl_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))

    # 过滤掉应该跳过的文件（必须以'g'开头，且不是应跳过的ch01文件）
    pred_kinetic_dir = os.path.join(predicted_pkl_root, 'kinetic_features')
    pred_manual_dir = os.path.join(predicted_pkl_root, 'manual_features_new')
    gt_kinetic_dir = os.path.join(gt_pkl_root, 'kinetic_features')
    gt_manual_dir = os.path.join(gt_pkl_root, 'manual_features_new')
    
    pred_features_k = [np.load(os.path.join(pred_kinetic_dir, pkl)) 
                       for pkl in os.listdir(pred_kinetic_dir) 
                       if should_process_file(pkl, pred_kinetic_dir)]
    pred_features_m = [np.load(os.path.join(pred_manual_dir, pkl)) 
                       for pkl in os.listdir(pred_manual_dir) 
                       if should_process_file(pkl, pred_manual_dir)]
    
    gt_freatures_k = [np.load(os.path.join(gt_kinetic_dir, pkl)) 
                      for pkl in os.listdir(gt_kinetic_dir) 
                      if should_process_file(pkl, gt_kinetic_dir)]
    gt_freatures_m = [np.load(os.path.join(gt_manual_dir, pkl)) 
                      for pkl in os.listdir(gt_manual_dir) 
                      if should_process_file(pkl, gt_manual_dir)]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 

#   T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize(gt_freatures_k)
    # gt_freatures_m = normalize(gt_freatures_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_g': fid_m, 'div_k': div_k, 'div_g' : div_m}
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root):
    
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []

    for pkl in os.listdir(root):
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        
        # 检查文件是否应该处理（必须以'g'开头，且不是应跳过的ch01文件）
        if not should_process_file(pkl, root):
            print(f"Skipping {pkl} (doesn't start with 'g' or is ch01 with ch02)")
            continue
        
        print(f"Processing {pkl}")
        # newname = os.path.splitext(os.path.basename(pkl))[0]
        
        # joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:1200,:]
        try:
            joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:1200,:]
        except:
            joint3d = np.load(os.path.join(root, pkl)).reshape(-1, 72)
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 24))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # np_dance[:, :3] = root
        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
        np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))


if __name__ == '__main__':


    gt_root = './data/aist_features_zero_start'
    pred_root = '/network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text/experiments/cc_motion_gpt_text_fix_choreo_aist_maintable/choreo_complete/twostage_nocot/ep000135'
    print('Calculating and saving features')
    # calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)


    print('Calculating metrics')
    print(gt_root)
    print(pred_root)
    print(quantized_metrics(pred_root, gt_root))