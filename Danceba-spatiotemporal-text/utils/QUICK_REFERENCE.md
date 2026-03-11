# metrics_new_finedance.py 修改快速参考

## 核心修改

### 1. 数据处理流程

#### GT数据 (315维 SMPLH)
```
[T, 315] (30fps, Y-up)
    ↓ process_gt_smplh_315()
    │ • 分离: root_pos(3) + rotation_6d(312)【无foot_contact】
    │ • 6D → rotation matrix → axis-angle
    │ • SMPLH FK: joints[T, 24, 3]（只用前24个关节rotation）
    │ • 截取前22个关节
    ↓
[T, 22, 3]
    ↓ apply_lodge_processing()
    │ • Step1: 减去第一帧root位置
    │ • Step2: 每帧其他关节减去当前帧root
    ↓
[T, 22, 3] → 特征提取
```

#### Pred数据 (keypoints3d)
```
[T, 72] or [T, 24, 3] (60fps, Y-up)
    ↓ process_pred_keypoints3d_60fps()
    │ • 统一reshape为[T, 24, 3]
    │ • 降采样60fps→30fps: data[::2]
    │ • 截取前22个关节
    ↓
[T/2, 22, 3]
    ↓ apply_lodge_processing()
    │ • 同GT处理
    ↓
[T/2, 22, 3] → 特征提取
```

---

## 2. 与LODGE的一致性

### ✅ 关节数量
- **之前**: 24个关节
- **现在**: 22个关节（与LODGE一致）

### ✅ root位置处理
- **之前**: 只减去第一帧root（1步）
- **现在**: 减去第一帧root + 每帧减去当前帧root（2步，与LODGE一致）

```python
# LODGE标准处理
joint3d_flat = joint3d.reshape(T, 66)
roott = joint3d_flat[:1, :3]
joint3d_flat = joint3d_flat - np.tile(roott, (1, 22))  # Step 1
joint3d = joint3d_flat.reshape(T, 22, 3)
joint3d[:, 1:, :] = joint3d[:, 1:, :] - joint3d[:, 0:1, :]  # Step 2
```

---

## 3. 主要新增函数

### `process_gt_smplh_315(data, smpl_model)`
- **输入**: [T, 315] = [3 root + 312 rotation]（无foot_contact）
- **输出**: [T, 22, 3]
- **用途**: GT数据SMPLH FK转关节位置（只使用前24个关节的rotation）

### `process_pred_keypoints3d_60fps(data)`
- **输入**: [T, 72] or [T, 24, 3], 60fps
- **输出**: [T/2, 22, 3], 30fps
- **用途**: Pred数据降采样+截取关节

### `apply_lodge_processing(joint3d)`
- **输入**: [T, 22, 3]
- **输出**: [T, 22, 3]
- **用途**: LODGE标准预处理（2步减root）

---

## 4. 使用方法

### 安装依赖
```bash
pip install torch smplx pytorch3d numpy scipy
```

### 准备SMPL模型
下载 SMPLH 模型并放置到 `./smpl/` 目录，或使用SMPL模型：
```python
# 推荐：使用SMPLH模型（支持52个关节）
from smplx import SMPLH
smpl_model = SMPLH(model_path='./smpl', gender='MALE')

# 或者：使用SMPL模型（只处理前24个关节）
from smplx import SMPL
smpl_model = SMPL(model_path='./smpl/SMPL_MALE.pkl', gender='male')
```

### 运行
```bash
cd /network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text/utils
python metrics_new_finedance.py
```

---

## 5. 数据要求检查

### GT数据
- ✅ 格式: `.npy`
- ✅ 形状: `[T, 315]` = `[3 root_pos + 312 rotation_6d]`
- ✅ 内容: SMPLH参数（52关节），**无foot_contact**
- ✅ 帧率: 30fps
- ✅ 坐标系: Y-up

### Pred数据
- ✅ 格式: `.npy`
- ✅ 形状: `[T, 72]` 或 `[T, 24, 3]`
- ✅ 内容: keypoints3d
- ✅ 帧率: 60fps
- ✅ 坐标系: Y-up

---

## 6. 调试提示

### 检查数据格式
```python
import numpy as np

# GT
gt = np.load('gt_file.npy')
print(f"GT shape: {gt.shape}")  # 应该是 (T, 315)

# Pred
pred = np.load('pred_file.npy')
print(f"Pred shape: {pred.shape}")  # 应该是 (T, 72) 或 (T, 24, 3)
```

### 验证帧率转换
```python
# Pred降采样后应该帧数减半
original_T = pred.shape[0]  # 例如 1200帧 (60fps)
processed = process_pred_keypoints3d_60fps(pred)
print(f"Original: {original_T}, Processed: {processed.shape[0]}")
# 应该输出: Original: 1200, Processed: 600
```

### 验证关节数
```python
joint3d = process_xxx(data)
print(f"Joint count: {joint3d.shape[1]}")  # 应该是 22
```

---

## 7. 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `AssertionError: Expected 315 dims` | GT数据不是315维 | 检查GT数据格式 |
| `FileNotFoundError: SMPL_MALE.pkl` | 缺少SMPL模型 | 下载并设置正确路径 |
| `ImportError: No module named 'smplx'` | 缺少依赖 | `pip install smplx` |
| `Expected 72 or 66 dims` | Pred数据格式不对 | 检查Pred数据是否为keypoints3d |

---

## 8. 输出特征

### Kinetic Features
- **维度**: 66 = 22关节 × 3指标
- **文件**: `kinetic_features/*.npy`

### Manual Features
- **维度**: 32
- **文件**: `manual_features_new/*.npy`

### Metrics
- **FID_k**: Kinetic FID
- **FID_g**: Geometric (Manual) FID
- **Div_k**: Kinetic Diversity
- **Div_g**: Geometric Diversity
