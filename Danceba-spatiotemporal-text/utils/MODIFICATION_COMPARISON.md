# metrics_new_finedance.py 修改前后对比

## 关键改变对比表

| 项目 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| **关节数量** | 24个 | 22个 | 与LODGE保持一致 |
| **GT数据处理** | 直接按keypoints处理 | 315维SMPLH → FK → keypoints | 正确处理SMPLH格式 |
| **Pred帧率** | 未处理 | 60fps → 30fps降采样 | 与GT帧率统一 |
| **Root处理步骤** | 1步（减第一帧） | 2步（减第一帧+减当前帧） | 符合LODGE标准 |

---

## 代码对比

### 1. 导入部分

#### 修改前
```python
import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import os
```

#### 修改后
```python
import numpy as np
import pickle 
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import os
import torch
from smplx import SMPL
```

**变化**: 新增 `torch` 和 `smplx` 导入

---

### 2. calc_and_save_feats 函数

#### 修改前
```python
def calc_and_save_feats(root):
    # ... 创建目录 ...
    
    for pkl in os.listdir(root):
        # ... 过滤逻辑 ...
        
        try:
            joint3d = np.load(...).item()['pred_position'][:1200,:]
        except:
            joint3d = np.load(...).reshape(-1, 72)[:1200,:]
        
        # 只减去第一帧root（1步）
        roott = joint3d[:1, :3]
        joint3d = joint3d - np.tile(roott, (1, 24))  # 24个关节
        
        # 直接提取特征
        np.save(..., extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
        np.save(..., extract_manual_features(joint3d.reshape(-1, 24, 3)))
```

#### 修改后
```python
def calc_and_save_feats(root, is_gt=False, smpl_model=None):
    # ... 创建目录 ...
    
    for pkl in os.listdir(root):
        # ... 过滤逻辑 ...
        
        data = np.load(os.path.join(root, pkl), allow_pickle=True)
        
        # 根据is_gt标志选择处理函数
        if is_gt:
            # GT: 315维SMPLH → FK → 22关节
            joint3d = process_gt_smplh_315(data, smpl_model)
        else:
            # Pred: 60fps → 30fps, 24关节 → 22关节
            joint3d = process_pred_keypoints3d_60fps(data)
        
        # 截取到1200帧
        if joint3d.shape[0] > 1200:
            joint3d = joint3d[:1200, :, :]
        
        # LODGE标准预处理（2步）
        joint3d = apply_lodge_processing(joint3d)
        
        # 提取特征
        np.save(..., extract_kinetic_features(joint3d))
        np.save(..., extract_manual_features(joint3d))
```

**主要变化**:
1. 新增 `is_gt` 和 `smpl_model` 参数
2. GT和Pred分别处理
3. 统一应用 `apply_lodge_processing()`
4. 使用22个关节而非24个

---

### 3. Root位置处理

#### 修改前（1步）
```python
roott = joint3d[:1, :3]  # 第一帧root
joint3d = joint3d - np.tile(roott, (1, 24))  # 所有关节减去第一帧root
# 结束
```

#### 修改后（2步，LODGE标准）
```python
def apply_lodge_processing(joint3d):
    # Step 1: 减去第一帧root位置
    joint3d_flat = joint3d.reshape(joint3d.shape[0], -1)  # [T, 66]
    roott = joint3d_flat[:1, :3]
    joint3d_flat = joint3d_flat - np.tile(roott, (1, 22))
    
    # Step 2: 每帧其他关节减去当前帧root
    joint3d = joint3d_flat.reshape(-1, 22, 3)
    joint3d[:, 1:, :] = joint3d[:, 1:, :] - joint3d[:, 0:1, :]
    
    return joint3d
```

**关键差异**: 
- 修改前只做了全局归一化（减第一帧）
- 修改后做了全局归一化+局部归一化（每帧相对root）

---

### 4. 主函数

#### 修改前
```python
if __name__ == '__main__':
    gt_root = './data/finedance_features_zero_start'
    pred_root = '...'
    
    print('Calculating and saving features')
    calc_and_save_feats(gt_root)  # GT和Pred一样处理
    calc_and_save_feats(pred_root)
    
    print('Calculating metrics')
    print(quantized_metrics(pred_root, gt_root))
```

#### 修改后
```python
if __name__ == '__main__':
    # 初始化SMPL模型
    smpl_model = SMPL(model_path='./smpl/SMPL_MALE.pkl', ...)
    
    gt_root = './data/finedance_features_zero_start'
    pred_root = '...'
    
    print('Calculating and saving features')
    
    # GT: 315维SMPLH，需要FK
    print('\n=== Processing GT data (315-dim SMPLH, 30fps) ===')
    calc_and_save_feats(gt_root, is_gt=True, smpl_model=smpl_model)
    
    # Pred: keypoints3d，需要降采样
    print('\n=== Processing Pred data (keypoints3d, 60fps) ===')
    calc_and_save_feats(pred_root, is_gt=False, smpl_model=None)
    
    print('\n=== Calculating metrics ===')
    metrics = quantized_metrics(pred_root, gt_root)
    print(metrics)
```

**主要变化**:
1. 初始化SMPL模型
2. GT和Pred分别处理，传入不同参数
3. 增加详细日志输出

---

## 处理流程对比图

### 修改前
```
GT + Pred (统一处理)
    ↓
加载数据 [T, 72]
    ↓
reshape为 [T, 24, 3]
    ↓
减去第一帧root (1步)
    ↓
特征提取 (24关节)
```

### 修改后
```
GT [T, 315, 30fps]          Pred [T, 72, 60fps]
    ↓                            ↓
SMPLH FK                     降采样 60→30fps
(只用前24关节rotation)           ↓
    ↓                        [T/2, 22, 3]
[T, 22, 3]                       ↓
    ↓                            ↓
    └──────┬──────┘
           ↓
    apply_lodge_processing()
    (减第一帧root + 每帧减当前root)
           ↓
      [T, 22, 3]
           ↓
      特征提取 (22关节)
```

---

## 数学上的差异

### 修改前
对于关节位置 $J \in \mathbb{R}^{T \times 24 \times 3}$:

$$J' = J - J[0, 0, :]$$

其中 $J[0, 0, :]$ 是第一帧的root位置

### 修改后（LODGE标准）
$$J' = J - J[0, 0, :]$$  (Step 1)
$$J'[t, j>0, :] = J'[t, j, :] - J'[t, 0, :]$$ (Step 2)

其中:
- Step 1: 全局归一化
- Step 2: 局部归一化（相对每帧root）

---

## 为什么需要这些修改？

### 1. 关节数量：24 → 22
**原因**: LODGE使用22个身体关节，排除了手部关节
**影响**: 特征维度从72变为66 (kinetic)

### 2. 增加第二步root处理
**原因**: LODGE的特征提取基于相对root的局部坐标
**影响**: 
- 运动不受全局平移影响
- 特征更关注姿态变化而非位置变化

### 3. GT数据单独处理
**原因**: GT是315维SMPLH参数，需要FK转换
**影响**: 能正确处理rotation参数

### 4. Pred降采样
**原因**: Pred是60fps，需要与GT的30fps对齐
**影响**: 时间对齐，确保公平比较

---

## 验证正确性

### 测试点1: 关节数量
```python
joint3d = process_xxx(data)
assert joint3d.shape[1] == 22, "Should have 22 joints"
```

### 测试点2: 帧率
```python
pred_60fps = np.load('pred.npy')  # [1200, 72], 60fps
joint3d = process_pred_keypoints3d_60fps(pred_60fps)
assert joint3d.shape[0] == 600, "Should be 30fps (half frames)"
```

### 测试点3: Root归一化
```python
joint3d = apply_lodge_processing(joint3d)
# 第一帧root应该为0
assert np.allclose(joint3d[0, 0, :], 0), "First frame root should be zero"
# 每帧root应该为0
assert np.allclose(joint3d[:, 0, :], 0), "All frame roots should be zero"
```

---

## 参考

- **LODGE代码**: https://github.com/li-ronghui/LODGE
- **详细文档**: `METRICS_MODIFICATION_SUMMARY.md`
- **快速参考**: `QUICK_REFERENCE.md`
