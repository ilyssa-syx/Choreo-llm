# SMPL/SMPLH 数据格式详解

## 快速理解：315维 vs 319维

### 可视化对比

```
315维格式：
┌───────────┬────────────────────────────────────┐
│  Root Pos │      Rotation (6D format)         │
│   (3维)   │          (312维)                   │
│  [0-2]    │         [3-314]                    │
└───────────┴────────────────────────────────────┘
     ↓
     └─ 没有 foot contact!

319维格式：
┌──────────┬───────────┬────────────────────────────────────┐
│   Foot   │  Root Pos │      Rotation (6D format)         │
│  Contact │   (3维)   │          (312维)                   │
│  (4维)   │  [4-6]    │         [7-318]                    │
│  [0-3]   │           │                                    │
└──────────┴───────────┴────────────────────────────────────┘
     ↑
     └─ 有 foot contact! (左右脚踝+脚尖的接地标志)

关系：319 = 315 + 4维foot_contact前缀
```

## 详细说明

### 315维格式（你的GT数据）

**结构**: `[3 root_pos + 312 rotation_6d]`

```python
data.shape = [T, 315]

# 维度划分
root_pos = data[:, 0:3]      # 前3维：根节点全局位置 (X, Y, Z)
rotation_6d = data[:, 3:315] # 后312维：52个关节的6D rotation

# 每个关节的6D rotation
# 312维 = 52关节 × 6维
# 关节0-21: 身体关节（SMPL的22个关节）
# 关节22-51: 手部关节（SMPLH的30个手指关节）
```

**特点**:
- ✅ 有root位置
- ✅ 有rotation信息
- ❌ **没有foot contact**（这是与319维的唯一区别！）

### 319维格式

**结构**: `[4 foot_contact + 3 root_pos + 312 rotation_6d]`

```python
data.shape = [T, 319]

# 维度划分
foot_contact = data[:, 0:4]   # 前4维：脚接地标志
                               # [左脚踝, 右脚踝, 左脚尖, 右脚尖]
root_pos = data[:, 4:7]       # 中3维：根节点全局位置
rotation_6d = data[:, 7:319]  # 后312维：52个关节的6D rotation
```

**特点**:
- ✅ 有foot contact
- ✅ 有root位置
- ✅ 有rotation信息

## Foot Contact 是什么？

Foot contact 是一个4维的二值向量，表示哪只脚接触地面：

```python
foot_contact = [left_ankle, right_ankle, left_toe, right_toe]

# 例子
[1, 0, 1, 0]  # 左脚着地，右脚抬起
[0, 1, 0, 1]  # 右脚着地，左脚抬起
[1, 1, 1, 1]  # 双脚着地
[0, 0, 0, 0]  # 双脚离地（跳跃）
```

**作用**:
- 可以用于脚滑（foot skating）检测
- 可以用于接地约束
- 但**不是必需的**，LODGE主要依赖FK后的关节位置

## LODGE如何处理315维？

LODGE代码中的处理：

```python
# LODGE的兼容性处理
if data.shape[1] == 315:
    # 在前面添加4维零（全零的foot contact）
    foot_contact = torch.zeros([data.shape[0], 4])
    data = torch.cat([foot_contact, data], dim=1)
    # 现在 data.shape = [T, 319]
```

所以你的315维GT数据会：
1. 自动在前面加4维零
2. 变成319维
3. 然后按319维的方式处理

## 你的case：metrics_new_finedance.py

在你的代码中，我们直接处理315维：

```python
def process_gt_smplh_315(data, smpl_model):
    """315维 -> SMPL FK -> 22个关节的3D位置"""
    
    # 直接从维度0开始就是root_pos（没有foot contact）
    root_pos = data[:, 0:3]      # 维度 0-2
    rotation_6d = data[:, 3:315] # 维度 3-314
    
    # 后续处理...
```

**不需要添加foot contact**，因为：
1. 我们直接进行FK转换
2. FK只需要root_pos和rotation
3. 最终的特征提取也不依赖foot contact

## 其他格式对比

### 139维 vs 135维（axis-angle格式）

同样的道理：

```
135维: [3 root_pos + 132 axis-angle]  ← 没有foot contact
139维: [4 foot_contact + 3 root_pos + 132 axis-angle]  ← 有foot contact

关系：139 = 135 + 4维foot_contact前缀
```

### Rotation表示方式对比

| 格式 | 每关节维度 | 52关节总维度 | 说明 |
|------|-----------|-------------|------|
| **Axis-angle** | 3维 | 156维 | 旋转轴×角度 |
| **6D Rotation** | 6维 | 312维 | Rotation matrix前两列 |
| **Quaternion** | 4维 | 208维 | 四元数（未使用） |

**为什么用6D？**
- 连续性好（没有奇异点）
- 易于神经网络优化
- 可直接转换为rotation matrix

## 总结

### 一句话记住

**315维 = 319维 - 4维foot_contact**

或者说：

**319维 = 315维 + 4维foot_contact前缀**

### 你的GT数据（315维）

```
结构: [3 root_pos + 312 rotation_6d]
特点: 没有foot contact，其他都有
处理: 直接提取root_pos和rotation，进行SMPL FK
```

### 如果是319维

```
结构: [4 foot_contact + 3 root_pos + 312 rotation_6d]
特点: 有foot contact
处理: 跳过前4维，提取root_pos和rotation，进行SMPL FK
```

### 关键点

✅ 315维和319维的rotation部分是**完全相同**的（都是312维）  
✅ 唯一区别是是否包含4维foot_contact前缀  
✅ SMPL FK不需要foot_contact，只需要root_pos和rotation  
✅ 你的代码已经正确处理了315维格式  
