
import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import scipy.interpolate

# kinetic, manual
import os
import torch
from smplx import SMPL

WHITE_LIST = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '124', '126', '128', '130', '132'}

def interpolate_to_60fps(data):
    """
    将30fps数据线性插值到60fps
    输入: [T, ...] - 任意形状的30fps数据
    输出: [T*2-1, ...] - 插值后的60fps数据
    """
    import scipy.interpolate
    
    T = data.shape[0]
    # 原始时间点（30fps）
    t_30fps = np.arange(T)
    # 目标时间点（60fps）
    t_60fps = np.linspace(0, T-1, 2*T-1)
    
    # 对每个维度进行插值
    if len(data.shape) == 2:
        # [T, D]
        interpolated = np.zeros((2*T-1, data.shape[1]))
        for i in range(data.shape[1]):
            f = scipy.interpolate.interp1d(t_30fps, data[:, i], kind='linear')
            interpolated[:, i] = f(t_60fps)
    elif len(data.shape) == 3:
        # [T, J, D]
        interpolated = np.zeros((2*T-1, data.shape[1], data.shape[2]))
        for j in range(data.shape[1]):
            for d in range(data.shape[2]):
                f = scipy.interpolate.interp1d(t_30fps, data[:, j, d], kind='linear')
                interpolated[:, j, d] = f(t_60fps)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    return interpolated


def process_gt_smplh_315(data, smpl_model):
    """
    处理GT数据：315维SMPLH格式，Y-up，30fps -> 60fps
    输入: [T, 315] - [3维root_pos + 312维rotation_6d (52关节×6)]，30fps
    注意: 315维格式不包含foot_contact，只有root_pos和rotation
    输出: [2T-1, 22, 3] - 插值到60fps后的22个身体关节的3D位置
    """
    assert data.shape[1] == 315, f"Expected 315 dims, got {data.shape[1]}"
    
    # 先插值到60fps
    print(f"  Interpolating GT from 30fps to 60fps: {data.shape[0]} -> {2*data.shape[0]-1} frames")
    data = interpolate_to_60fps(data)  # [T, 315] -> [2T-1, 315]
    
    # 分离各部分（315维没有foot_contact！）
    root_pos = data[:, :3]        # [2T-1, 3] - 维度0-2
    rotation_6d = data[:, 3:]     # [2T-1, 312] - 维度3-314，52关节 × 6
    
    T = data.shape[0]
    
    # 6D rotation转为rotation matrix再转为axis-angle
    # rotation_6d: [T, 312] -> [T, 52, 6]
    rotation_6d = rotation_6d.reshape(T, 52, 6)
    
    # 转换为torch tensor
    rotation_6d_torch = torch.from_numpy(rotation_6d).float()
    root_pos_torch = torch.from_numpy(root_pos).float()
    
    # 6D to rotation matrix
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
    rot_mat = rotation_6d_to_matrix(rotation_6d_torch)  # [T, 52, 3, 3]
    axis_angle = matrix_to_axis_angle(rot_mat)  # [T, 52, 3]
    
    # SMPLH forward kinematics - 只使用前24个身体关节的rotation
    # SMPLH有52个关节，但前24个是身体关节（与SMPL相同）
    global_orient = axis_angle[:, 0:1, :]  # [T, 1, 3]
    body_pose = axis_angle[:, 1:24, :]  # [T, 23, 3]
    
    # Forward kinematics（SMPLH也可以只用body_pose）
    smpl_output = smpl_model(
        global_orient=global_orient.reshape(T, 3),
        body_pose=body_pose.reshape(T, 23*3),
        transl=root_pos_torch,
        return_verts=False
    )
    
    # 获取关节位置并只保留前22个身体关节
    joint3d = smpl_output.joints.detach().cpu().numpy()[:, :22, :]  # [T, 22, 3]
    
    return joint3d


def process_pred_keypoints3d_60fps(data):
    """
    处理预测数据：keypoints3d格式，Y-up，60fps
    输入: [T, 72] 或 [T, 24, 3]，60fps
    输出: [T, 22, 3]，60fps（不降采样），只保留前22个关节
    """
    print(f"  Processing Pred data: keeping 60fps, shape={data.shape}")
    
    # 处理输入形状
    if len(data.shape) == 2:
        if data.shape[1] == 72:
            # [T, 72] -> [T, 24, 3]
            data = data.reshape(-1, 24, 3)
        elif data.shape[1] == 66:
            # [T, 66] -> [T, 22, 3] (已经是22个关节)
            data = data.reshape(-1, 22, 3)
            return data  # 直接返回，无需再截取关节
        else:
            raise ValueError(f"Expected 72 or 66 dims, got {data.shape[1]}")
    elif len(data.shape) == 3:
        # [T, J, 3]
        if data.shape[1] == 24:
            pass  # 保持不变
        elif data.shape[1] == 22:
            return data  # 直接返回
        else:
            raise ValueError(f"Expected 24 or 22 joints, got {data.shape[1]}")
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    # 只保留前22个身体关节（不降采样）
    joint3d = data[:, :22, :]  # [T, 22, 3]
    
    return joint3d


def apply_lodge_processing(joint3d):
    """
    应用LODGE的特征提取预处理：
    1. 减去第一帧的root位置（让动作从原点开始）
    2. 每帧的其他关节减去当前帧的root（相对root的位置）
    
    输入: [T, 22, 3]
    输出: [T, 22, 3]
    """
    # Step 1: 转为flat格式进行第一步处理
    joint3d_flat = joint3d.reshape(joint3d.shape[0], -1)  # [T, 66]
    
    # 减去第一帧的root位置
    roott = joint3d_flat[:1, :3]  # 第一帧的root位置 [1, 3]
    joint3d_flat = joint3d_flat - np.tile(roott, (1, 22))  # [T, 66]
    
    # Step 2: 转回3D格式，每帧的其他关节减去当前帧的root
    joint3d = joint3d_flat.reshape(-1, 22, 3)  # [T, 22, 3]
    joint3d[:, 1:, :] = joint3d[:, 1:, :] - joint3d[:, 0:1, :]  # 其他关节相对于root
    
    return joint3d


def should_skip_ch01(name, base_dir):
    if 'ch01' not in name:
        return False
    alt = name.replace('ch01', 'ch02')
    # 如果存在对应的 ch02 文件，则跳过 ch01 文件
    if os.path.exists(os.path.join(base_dir, alt)):
        return True
    return False

def should_process_file(name, base_dir):
    """检查文件是否应该参与计算（只处理不以'g'开头的文件）"""
    # 文件名第一个字母不能是 'g'
    if name.startswith('g'):
        return False
    return True

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    # 对std≈0的维度（GT样本在该维度上取值恒定），不做除法放大
    std = np.where(std < 1e-6, 1.0, std)
    return (feat - mean) / std, (feat2 - mean) / std

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

    # 过滤掉应该跳过的文件（只处理不以'g'开头的文件，且不是应跳过的ch01文件）
    pred_kinetic_dir = os.path.join(predicted_pkl_root, 'kinetic_features')
    pred_manual_dir = os.path.join(predicted_pkl_root, 'manual_features_new')
    gt_kinetic_dir = os.path.join(gt_pkl_root, 'kinetic_features')
    gt_manual_dir = os.path.join(gt_pkl_root, 'manual_features_new')
    
    pred_features_k = [np.load(os.path.join(pred_kinetic_dir, pkl)) 
                       for pkl in os.listdir(pred_kinetic_dir) 
                       if pkl.split('.')[0] in WHITE_LIST]
    pred_features_m = [np.load(os.path.join(pred_manual_dir, pkl)) 
                       for pkl in os.listdir(pred_manual_dir) 
                       if pkl.split('.')[0] in WHITE_LIST]
    
    gt_freatures_k = [np.load(os.path.join(gt_kinetic_dir, pkl)) 
                      for pkl in os.listdir(gt_kinetic_dir) ]
    gt_freatures_m = [np.load(os.path.join(gt_manual_dir, pkl)) 
                      for pkl in os.listdir(gt_manual_dir) ]
    print(len(pred_features_k), len(pred_features_m), len(gt_freatures_k), len(gt_freatures_m))
    
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

def calc_and_save_feats(root, is_gt=False, smpl_model=None):
    """
    计算并保存特征
    
    Args:
        root: 数据目录
        is_gt: 是否为groundtruth数据（315维SMPLH格式）
        smpl_model: SMPL模型（仅GT需要）
    """
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    for pkl in os.listdir(root):
        # 只处理.npy文件
        if not pkl.endswith('.npy'):
            continue
            
        # 检查文件是否应该处理
        if not should_process_file(pkl, root):
            print(f"Skipping {pkl} (starts with 'g' or is ch01 with ch02)")
            continue
        
        print(f"Processing {pkl}")
        
        try:
            # 加载数据
            data = np.load(os.path.join(root, pkl), allow_pickle=True)
            
            # 根据是GT还是Pred进行不同处理
            if is_gt:
                # GT: 315维SMPLH格式，30fps
                if isinstance(data, np.ndarray) and data.shape[1] == 315:
                    joint3d = process_gt_smplh_315(data, smpl_model)
                else:
                    print(f"Skipping {pkl}: GT data should be [T, 315], got shape {data.shape}")
                    continue
            else:
                # Pred: keypoints3d格式，60fps
                # 检查是否是0维数组（通常包含字典）
                if isinstance(data, np.ndarray) and data.shape == ():
                    # 0维数组，提取包装的对象（通常是字典）
                    print(f"  Data is 0-dim array, extracting wrapped object")
                    try:
                        data = data.item()
                        if isinstance(data, dict):
                            if 'pred_position' in data:
                                joint3d = process_pred_keypoints3d_60fps(data['pred_position'])
                            else:
                                print(f"Skipping {pkl}: No 'pred_position' key in dict, keys: {data.keys()}")
                                continue
                        else:
                            print(f"Skipping {pkl}: 0-dim array contains {type(data)}, not dict")
                            continue
                    except Exception as e:
                        print(f"Skipping {pkl}: Error extracting from 0-dim array: {e}")
                        continue
                elif isinstance(data, np.ndarray):
                    # 正常的多维numpy数组
                    print(f"  Data shape: {data.shape}")
                    if len(data.shape) == 2 and data.shape[1] in [72, 66]:  # [T, 72] or [T, 66]
                        joint3d = process_pred_keypoints3d_60fps(data)
                    elif len(data.shape) == 3 and data.shape[1] in [24, 22]:  # [T, 24, 3] or [T, 22, 3]
                        joint3d = process_pred_keypoints3d_60fps(data)
                    else:
                        print(f"Unexpected pred shape for {pkl}: {data.shape}")
                        continue
                else:
                    print(f"Skipping {pkl}: Unknown data type {type(data)}")
                    continue
            
            # 截取到2400帧（60fps下为40秒，与之前30fps的1200帧等效）
            if joint3d.shape[0] > 2400:
                joint3d = joint3d[:2400, :, :]
            
            # 应用LODGE的预处理
            joint3d = apply_lodge_processing(joint3d)
            
            # 提取并保存特征
            np.save(
                os.path.join(root, 'kinetic_features', pkl), 
                extract_kinetic_features(joint3d)
            )
            np.save(
                os.path.join(root, 'manual_features_new', pkl), 
                extract_manual_features(joint3d)
            )
            
        except Exception as e:
            print(f"Error processing {pkl}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    # 初始化SMPLH模型（用于GT数据的FK）
    # 注意：数据是315维SMPLH格式（52个关节），需要使用SMPLH模型
    # 但我们只使用前24个身体关节的rotation进行FK
    try:
        from smplx import SMPLH
        smpl_model = SMPLH(
            model_path='/network_space/server127_2/shared/sunyx3/tools/smpl/SMPLH_MALE.pkl',
            gender='MALE',
            use_pca=False,
            create_transl=False
        )
    except:
        # 如果没有SMPLH，尝试使用SMPL（只能处理前24个关节的rotation）
        print("Warning: SMPLH not found, using SMPL. This may work since we only use first 24 joints.")
        smpl_model = SMPL(
            model_path='/network_space/server127_2/shared/sunyx3/tools/smpl/SMPL_MALE.pkl',
            gender='male',
            create_transl=False
        )
    
    gt_root = './data/finedance_features_zero_start'
    # pred_root = '/network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text/experiments/cc_motion_gpt_text_fix_choreo_aist_maintable/choreo_complete/maintable/ep000135'
    pred_root = '/network_space/server126/shared/sunyx/models/Bailando/experiments/actor_critic/eval/pkl/ep000010'
    print('Calculating and saving features')
    print(f'GT root: {gt_root}')
    print(f'Pred root: {pred_root}')
    
    # GT数据：315维SMPLH格式，30fps -> 插值到60fps
    print('\n=== Processing GT data (315-dim SMPLH, 30fps -> 60fps interpolation) ===')
    calc_and_save_feats(gt_root, is_gt=True, smpl_model=smpl_model)
    
    # Pred数据：keypoints3d格式，60fps（保持不变，不降采样）
    print('\n=== Processing Pred data (keypoints3d, 60fps - no downsampling) ===')
    calc_and_save_feats(pred_root, is_gt=False, smpl_model=None)

    print('\n=== Calculating metrics ===')
    print(gt_root)
    print(pred_root)
    metrics = quantized_metrics(pred_root, gt_root)
    print(metrics)