# metrics_new_finedance.py 修改总结

## 修改日期
2026-03-06

## 修改目的
使 metrics 评估代码符合 LODGE 标准，正确处理两种不同格式的输入数据。

---

## 主要修改

### 1. 新增依赖
```python
import torch
from smplx import SMPL
```

### 2. 新增数据处理函数

#### 2.1 `process_gt_smplh_315(data, smpl_model)`
**功能**: 处理 Groundtruth 数据（315维 SMPLH 格式）

**输入**:
- `data`: `[T, 315]` - 包含 [3维root_pos + 312维rotation_6d (52关节×6)]，**无foot_contact**
- `smpl_model`: SMPLH 模型实例（或SMPL模型，只使用前24个关节）

**处理步骤**:
1. 分离 root_pos (3维), rotation_6d (312维)，**注意315维格式不包含foot_contact**
2. 将 6D rotation (312维) 转换为 rotation matrix，再转换为 axis-angle
3. 使用 SMPLH forward kinematics 计算关节位置（只使用前24个身体关节的rotation）
4. 只保留前 22 个身体关节

**输出**: `[T, 22, 3]` - 22个身体关节的3D位置，30fps

---

#### 2.2 `process_pred_keypoints3d_60fps(data)`
**功能**: 处理预测数据（keypoints3d 格式，60fps）

**输入**:
- `data`: `[T, 72]` 或 `[T, 24, 3]`，60fps

**处理步骤**:
1. 统一输入格式为 `[T, 24, 3]`
2. 从 60fps 降采样到 30fps：`data[::2]`
3. 只保留前 22 个身体关节

**输出**: `[T/2, 22, 3]` - 22个身体关节的3D位置，30fps

---

#### 2.3 `apply_lodge_processing(joint3d)`
**功能**: 应用 LODGE 标准的特征提取预处理

**输入**: `[T, 22, 3]`

**处理步骤**（与 LODGE 完全一致）:
1. **第一步**: 减去第一帧的 root 位置
   ```python
   joint3d_flat = joint3d.reshape(T, 66)
   roott = joint3d_flat[:1, :3]  # 第一帧的root
   joint3d_flat = joint3d_flat - np.tile(roott, (1, 22))
   ```

2. **第二步**: 每帧的其他关节减去当前帧的 root
   ```python
   joint3d = joint3d_flat.reshape(T, 22, 3)
   joint3d[:, 1:, :] = joint3d[:, 1:, :] - joint3d[:, 0:1, :]
   ```

**输出**: `[T, 22, 3]` - 处理后的相对关节位置

---

### 3. 修改 `calc_and_save_feats()` 函数

**新增参数**:
- `is_gt`: 布尔值，标识是否为 groundtruth 数据
- `smpl_model`: SMPL 模型实例（仅 GT 需要）

**主要改进**:
1. 根据 `is_gt` 标志使用不同的处理函数
2. GT 数据调用 `process_gt_smplh_315()`
3. Pred 数据调用 `process_pred_keypoints3d_60fps()`
4. 统一应用 `apply_lodge_processing()` 进行 LODGE 标准预处理
5. 增强错误处理和日志输出

**处理流程**:
```
GT:  [T, 315] -> process_gt_smplh_315() -> [T, 22, 3] -> apply_lodge_processing() -> 特征提取
Pred: [T, 72] or [T, 24, 3], 60fps -> process_pred_keypoints3d_60fps() -> [T/2, 22, 3], 30fps -> apply_lodge_processing() -> 特征提取
```

---

### 4. 修改主函数 `if __name__ == '__main__'`

**主要改进**:
1. 初始化 SMPL 模型用于 GT 数据的 FK
2. 分别处理 GT 和 Pred 数据，传入正确的参数
3. 增加详细的日志输出

**代码结构**:
```python
# 初始化SMPL模型
smpl_model = SMPL(model_path='./smpl/SMPL_MALE.pkl', ...)

# 处理GT数据（315维SMPLH，30fps）
calc_and_save_feats(gt_root, is_gt=True, smpl_model=smpl_model)

# 处理Pred数据（keypoints3d，60fps）
calc_and_save_feats(pred_root, is_gt=False, smpl_model=None)

# 计算metrics
metrics = quantized_metrics(pred_root, gt_root)
```

---

## 与 LODGE 的一致性

### ✅ 数据格式
- GT: 315维 SMPLH -> 通过 FK 转为关节位置
- Pred: keypoints3d (已经是关节位置)
- 统一输出: 22个身体关节的3D位置

### ✅ 帧率处理
- GT: 保持 30fps（无需调整）
- Pred: 从 60fps 降采样到 30fps

### ✅ 坐标系
- 统一使用 Y-up 坐标系（高度轴是 Y）

### ✅ 特征提取预处理
完全遵循 LODGE 的两步处理：
1. 减去第一帧 root 位置（让动作从原点开始）
2. 每帧的其他关节减去当前帧 root（相对 root 的位置）

这与 LODGE 代码完全一致：
```python
# LODGE 的处理方式
roott = joint3d[:1, :3]
joint3d = joint3d - np.tile(roott, (1, 22))
joint3d = joint3d.reshape(-1, 22, 3)
joint3d[:, 1:, :] = joint3d[:, 1:, :] - joint3d[:, 0:1, :]
```

---

## 关键差异对比

### 之前的代码
```python
# 只处理24个关节
joint3d = data.reshape(-1, 72)[:1200,:]
roott = joint3d[:1, :3]
joint3d = joint3d - np.tile(roott, (1, 24))  # 只有第一步
# 缺少第二步：每帧减去当前帧root
```

### 修改后的代码
```python
# 只处理22个关节（符合LODGE）
joint3d = process_xxx()  # 统一转为[T, 22, 3]
joint3d = apply_lodge_processing(joint3d)  # 两步处理都有
```

---

## 使用说明

### 前提条件
1. 安装依赖：`pip install torch smplx pytorch3d`
2. 准备 SMPLH 模型文件：下载SMPLH模型到 `./smpl/` 目录
   - 或使用 SMPL 模型（只处理前24个关节的rotation）

### 运行
```bash
cd /network_space/server126/shared/sunyx/models/Danceba-spatiotemporal-text/utils
python metrics_new_finedance.py
```

### 数据要求
- **GT 数据**: `.npy` 文件，形状 `[T, 315]` = `[3 root + 312 rotation]`（**无foot_contact**），30fps，Y-up
- **Pred 数据**: `.npy` 文件，形状 `[T, 72]` 或 `[T, 24, 3]`，60fps，Y-up

---

## 输出特征

### Kinetic Features
- 维度: 66 (22关节 × 3个指标)
- 内容: 水平动能、垂直动能、能量消耗

### Manual Features  
- 维度: 32
- 内容: 关节间距离、平面关系、角度、速度

---

## 测试建议

### 1. 验证数据格式
```python
# GT数据
gt_data = np.load('gt_file.npy')
assert gt_data.shape[1] == 315, "GT should be 315-dim"

# Pred数据
pred_data = np.load('pred_file.npy')
assert pred_data.shape[1] in [72, 66] or pred_data.shape == (T, 24, 3)
```

### 2. 验证帧率转换
```python
# Pred从60fps降到30fps后帧数应该减半
original_frames = pred_data.shape[0]
processed = process_pred_keypoints3d_60fps(pred_data)
assert processed.shape[0] == original_frames // 2
```

### 3. 验证关节数量
```python
# 最终都应该是22个关节
assert joint3d.shape[1] == 22
```

---

## 可能的问题与解决方案

### 问题1: SMPL模型路径错误
**错误**: `FileNotFoundError: ./smpl/SMPL_MALE.pkl`

**解决**: 修改 `model_path` 为实际路径，或从 [SMPL官网](https://smpl.is.tue.mpg.de/) 下载模型

### 问题2: GT数据不是315维
**错误**: `AssertionError: Expected 315 dims, got XXX`

**解决**: 检查 GT 数据格式，确认是 SMPLH 格式（315维）

### 问题3: 内存不足
**错误**: `RuntimeError: CUDA out of memory`

**解决**: 
- 减少 batch_size
- 使用 CPU 版本：`smpl_model = SMPL(...).cpu()`
- 处理分块：每次只处理部分数据

---

## 参考资料

1. **LODGE Repository**: https://github.com/li-ronghui/LODGE
2. **LODGE Metrics文档**: `LODGE_metrics_input_requirements.md`
3. **SMPL/SMPLX**: https://smpl.is.tue.mpg.de/
4. **PyTorch3D**: https://pytorch3d.org/
