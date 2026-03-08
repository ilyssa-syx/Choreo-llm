# Gemini 回复域检查和重试机制

## 概述

为 `call_gemini.py` 添加了自动检查 Gemini API 回复是否包含必需域的功能。如果回复缺少必需的域，会自动重试调用 API，最多重试 5 次（可配置）。

## 主要功能

### 1. 域检查函数 `check_response_domains()`

```python
def check_response_domains(response_text, required_domains=None):
    """
    检查Gemini回复是否包含所有必需的域（domain）
    
    Args:
        response_text: Gemini的回复文本
        required_domains: 必需的域列表，默认为 ['whole body', 'lower half body', 'upper half body', 'torso']
    
    Returns:
        (is_valid, missing_domains): is_valid为True表示所有域都存在，missing_domains为缺失的域列表
    """
```

**特点：**
- 默认检查 4 个域：`whole body`、`lower half body`、`upper half body`、`torso`
- 支持自定义域列表，方便将来扩展
- 不区分大小写（例如 `**whole body**` 和 `**WHOLE BODY**` 都能识别）
- 返回缺失的域列表，便于调试

### 2. 自动重试机制

修改了 `process_single_segment()` 函数，添加了自动重试逻辑：

- **最多重试 5 次**（可配置）
- 每次调用 API 后立即检查域的完整性
- 如果所有域都存在，立即返回结果，不再重试
- 如果达到最大重试次数仍有缺失，会在回复前添加警告信息

### 3. 命令行参数

添加了两个新的命令行参数：

#### `--max_retries`
- 类型：整数
- 默认值：5
- 说明：API 调用最大重试次数

#### `--required_domains`
- 类型：字符串（逗号分隔）
- 默认值：None（使用默认的 4 个域）
- 说明：必需的域列表，用逗号分隔
- 示例：`--required_domains "whole body,lower half body,upper half body,torso,head"`

## 使用方法

### 基本使用（使用默认配置）

```bash
python call_gemini.py \
    --json_folder ./json_files \
    --video_folder ./videos \
    --output_folder ./results \
    --prompt_file ./prompt.txt \
    --modifier_folder ./modifiers
```

**默认行为：**
- 检查 4 个默认域
- 最多重试 5 次

### 自定义重试次数

```bash
python call_gemini.py \
    --json_folder ./json_files \
    --video_folder ./videos \
    --output_folder ./results \
    --prompt_file ./prompt.txt \
    --max_retries 10
```

### 自定义必需的域

```bash
python call_gemini.py \
    --json_folder ./json_files \
    --video_folder ./videos \
    --output_folder ./results \
    --prompt_file ./prompt.txt \
    --required_domains "whole body,lower half body,torso"
```

### 完整配置示例

```bash
python call_gemini.py \
    --json_folder ./json_files \
    --video_folder ./videos \
    --output_folder ./results \
    --prompt_file ./prompt.txt \
    --modifier_folder ./modifiers \
    --max_workers 10 \
    --max_json_workers 3 \
    --max_retries 8 \
    --required_domains "whole body,lower half body,upper half body,torso,head"
```

## 输出示例

### 成功的情况

```
Segment 0: 已创建完成full_prompt！
Segment 0: 已创建完成video_bytes！
Segment 0: 完成第1次API调用！
Segment 0: 验证通过，所有域都存在！
[1/5] ✓ Segment 0 处理完成
```

### 需要重试的情况

```
Segment 1: 已创建完成full_prompt！
Segment 1: 已创建完成video_bytes！
Segment 1: 完成第1次API调用！
Segment 1: 第1次尝试缺少域: ['torso']
Segment 1: 完成第2次API调用！
Segment 1: 验证通过，所有域都存在！
[2/5] ✓ Segment 1 处理完成
```

### 达到最大重试次数的情况

```
Segment 2: 已创建完成full_prompt！
Segment 2: 已创建完成video_bytes！
Segment 2: 完成第1次API调用！
Segment 2: 第1次尝试缺少域: ['upper half body']
Segment 2: 完成第2次API调用！
Segment 2: 第2次尝试缺少域: ['upper half body']
...
Segment 2: 完成第5次API调用！
Segment 2: 第5次尝试缺少域: ['upper half body']
Segment 2: [警告] 经过5次尝试仍缺少域: ['upper half body']
[3/5] ✓ Segment 2 处理完成
```

**注意：** 即使达到最大重试次数，仍会保存最后一次的回复，并在回复前添加警告信息。

## 如何添加新的域

有两种方法：

### 方法 1: 使用命令行参数（推荐）

直接在命令行中指定所有需要的域：

```bash
python call_gemini.py \
    --json_folder ./json_files \
    --video_folder ./videos \
    --output_folder ./results \
    --prompt_file ./prompt.txt \
    --required_domains "whole body,lower half body,upper half body,torso,head,left arm,right arm"
```

### 方法 2: 修改默认值

如果希望永久修改默认域列表，可以编辑 `call_gemini.py` 文件，找到 `check_response_domains()` 函数：

```python
def check_response_domains(response_text, required_domains=None):
    if required_domains is None:
        # 在这里修改默认域列表
        required_domains = ['whole body', 'lower half body', 'upper half body', 'torso', 'head']
    # ...
```

## 注意事项

1. **域名格式：** 函数查找的是 `**domain**` 格式（Markdown 加粗），不区分大小写
2. **重试成本：** 每次重试都会调用 API，会增加 API 调用次数和费用
3. **警告信息：** 达到最大重试次数后的警告会追加到 JSON 输出中，方便后续筛选和处理
4. **视频字节复用：** 视频字节数据只创建一次，多次重试时复用同一份数据，节省处理时间

## 测试

提供了测试脚本 `test_domain_check.py` 来验证域检查功能：

```bash
python test_domain_check.py
```

测试覆盖：
- 完整回复（包含所有域）
- 不完整回复（缺少某些域）
- 自定义域列表
- 空回复
- 大小写混合

## 性能影响

- **最佳情况：** 回复完整，无重试，性能与原代码相同
- **平均情况：** 1-2 次重试，增加少量 API 调用时间
- **最坏情况：** 5 次重试，时间增加约 5 倍（但大多数情况下不会发生）

## 兼容性

- 完全向后兼容：不传递新参数时使用默认值
- 所有现有功能保持不变
- 不影响已有的并发处理逻辑
