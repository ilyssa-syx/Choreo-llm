import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)
from smplx import SMPL
import os



FPS = 30

def apply_aist_rotation(smpl_poses, smpl_trans):
    """
    Apply AIST++ coordinate system rotation (y-up to z-up)
    """
    # Convert to torch tensors
    root_pos = torch.Tensor(smpl_trans)
    local_q = torch.Tensor(smpl_poses)
    
    # Reshape local_q to (frames, joints, 3)
    sq, c = local_q.shape
    local_q = local_q.reshape((sq, -1, 3))

    # AISTPP dataset comes y-up - rotate to z-up to standardize
    root_q = local_q[:, :1, :]  # sequence x 1 x 3 (root joint only)
    root_q_quat = axis_angle_to_quaternion(root_q)
    rotation = torch.Tensor([0.7071068, 0.7071068, 0, 0])  # 90 degrees about the x axis
    root_q_quat = quaternion_multiply(rotation, root_q_quat)
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :1, :] = root_q
    
    # Rotate the root position too
    pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
    root_pos = pos_rotation.transform_points(root_pos)  # (y, z) -> (-z, y)
    
    # Reshape back
    local_q = local_q.reshape((sq, c))
    
    # Convert back to numpy
    return local_q.numpy(), root_pos.numpy()


def get_smplinfo(data):
    if 'q' in data and 'pos' in data:
        smpl_poses = data['q']  # (150, 24, 3)
        smpl_trans = data['pos']  # (150, 3)
    else:
        smpl_poses = data['smpl_poses']  # (150, 24, 3)
        smpl_trans = data['smpl_trans']  # (150, 3)
    if 'scale' in data:
        smpl_scaling = data['scale']
    else:
        smpl_scaling = 1.0
    if 'q' in data and 'pos' in data and FPS==30:
        smpl_poses = smpl_poses[::2]
        smpl_trans = smpl_trans[::2]
    # print('scaling:', smpl_scaling)
    # print('smpl_poses shape: ',smpl_poses.shape)
    # print('smpl_trans shape: ',smpl_trans.shape)
    return smpl_poses, smpl_scaling, smpl_trans

def get_keypoints_from_smpl(smpl_data):
    smpl = SMPL(model_path='utils/smpl/SMPL_MALE.pkl', gender='MALE', batch_size=1)
    smpl_poses, smpl_scaling, smpl_trans = get_smplinfo(data=smpl_data)
    keypoints3d = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:3]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 3:]).float(),
        transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
        ).joints.detach().numpy()[:, 0:24, :]
    roott = keypoints3d[:1, :1]  # the root
    keypoints3d = keypoints3d - roott

    return keypoints3d

