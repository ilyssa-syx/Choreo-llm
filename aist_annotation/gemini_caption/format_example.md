# Modifier 格式化示例

## 旧格式（使用序数词）
```
* **First:** Initial standing position
* **Second:** Transition to forward lean
* **Third:** Forward lean with arm raise
* **Fourth:** Transition to side step
* **Fifth:** Final pose with arms extended
```

## 新格式（Pose/Trans 交替）
```
* **Pose 1:** Initial standing position
* **Trans 1:** Transition to forward lean
* **Pose 2:** Forward lean with arm raise
* **Trans 2:** Transition to side step
* **Pose 3:** Final pose with arms extended
```

## 规则说明

1. **偶数索引（0, 2, 4, ...）** → Pose 1, Pose 2, Pose 3, ...
2. **奇数索引（1, 3, 5, ...）** → Trans 1, Trans 2, Trans 3, ...
3. **Pose 总是比 Trans 多 1 个**（当修饰符数量为奇数时）

## 不同数量的示例

### 1个modifier（1 Pose + 0 Trans）
```
* **Pose 1:** Initial standing position
```

### 3个modifiers（2 Pose + 1 Trans）
```
* **Pose 1:** Initial standing position
* **Trans 1:** Transition to forward lean
* **Pose 2:** Forward lean with arm raise
```

### 5个modifiers（3 Pose + 2 Trans）
```
* **Pose 1:** Initial standing position
* **Trans 1:** Transition to forward lean
* **Pose 2:** Forward lean with arm raise
* **Trans 2:** Transition to side step
* **Pose 3:** Final pose with arms extended
```

### 6个modifiers（3 Pose + 3 Trans）
```
* **Pose 1:** Initial standing position
* **Trans 1:** Transition to forward lean
* **Pose 2:** Forward lean with arm raise
* **Trans 2:** Transition to side step
* **Pose 3:** Side step position
* **Trans 3:** Transition to final pose
```

## 说明

- **Pose** 表示关键姿态/位置
- **Trans** 表示姿态之间的过渡/转换
- 这种格式更符合舞蹈动作的描述逻辑（姿态 → 过渡 → 姿态 → 过渡 → ...）
