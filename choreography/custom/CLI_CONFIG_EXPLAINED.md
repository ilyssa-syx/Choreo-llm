# CLI 参数和配置系统详解

## 问题 1: ArgumentGroup 是什么？和 Parser 的关系？

### ArgumentParser 和 ArgumentGroup 的概念

```python
# 创建主解析器
parser = argparse.ArgumentParser(...)

# 创建参数组（仅用于帮助信息的分组显示）
search_grp = parser.add_argument_group("搜索参数")
filter_grp = parser.add_argument_group("元数据过滤")
vision_grp = parser.add_argument_group("视觉检测")
```

**关系图：**
```
ArgumentParser (parser)
    ├── 主参数 (--log_level, --log_file 等)
    └── 参数组 ArgumentGroup (仅用于帮助信息分类)
        ├── "搜索参数" 组
        │   ├── --queries_file
        │   ├── --max_pages_per_query
        │   └── --search_order
        ├── "元数据过滤" 组
        │   ├── --min_duration_sec
        │   └── --max_duration_sec
        └── "视觉检测" 组
            ├── --video_dir
            └── --sample_every_sec
```

### ArgumentGroup 的作用

**作用：仅用于组织帮助信息的显示，不影响功能**

```python
# cli.py 中的代码
search_grp = parser.add_argument_group("搜索参数")  # 创建一个显示组
search_grp.add_argument("--queries_file", ...)      # 往这个组里添加参数
search_grp.add_argument("--max_pages_per_query", ...)
```

**效果：当用户运行 `python -m bili_solo_pipeline.cli --help` 时**

```
usage: bili-solo-pipeline [options]

搜索参数:
  --queries_file PATH        关键词文件路径
  --max_pages_per_query N    每个关键词最多搜索页数
  --search_order {totalrank,click}  排序方式

元数据过滤:
  --min_duration_sec SEC     最短视频时长
  --max_duration_sec SEC     最长视频时长

视觉检测:
  --video_dir DIR            本地视频目录
  --sample_every_sec SEC     抽帧间隔
```

**重点：**
- `add_argument_group()` **只是为了将参数在帮助信息中分组显示**
- 功能上和直接 `parser.add_argument()` 完全一样
- 所有参数最终都是由同一个 `parser` 解析的

### 为什么要用 ArgumentGroup？

```python
# 不用 ArgumentGroup（所有参数混在一起）
parser.add_argument("--queries_file", ...)
parser.add_argument("--min_duration_sec", ...)
parser.add_argument("--video_dir", ...)
parser.add_argument("--solo_threshold", ...)
# 帮助信息：60+ 个参数全部堆在一起，用户懵逼 😵

# 使用 ArgumentGroup（分类清晰）
search_grp = parser.add_argument_group("搜索参数")
search_grp.add_argument("--queries_file", ...)
filter_grp = parser.add_argument_group("元数据过滤")
filter_grp.add_argument("--min_duration_sec", ...)
# 帮助信息：按功能分类，用户一目了然 ✅
```

---

## 问题 2: config.py 里面是什么？

### config.py 的三大部分

#### 1️⃣ 数据定义（舞种词表、黑白名单）

```python
# 舞种 -> 搜索关键词映射
GENRE_KEYWORD_MAP: dict[str, list[str]] = {
    "jazz": ["jazz", "jazz dance", "爵士舞"],
    "hiphop": ["hiphop", "hip-hop", "嘻哈舞"],
    "urban": ["urban", "urban dance", "urban编舞"],
    # ... 更多舞种
}

# 通用舞蹈词
GENERAL_DANCE_KEYWORDS = ["舞蹈", "编舞", "独舞", "solo"]

# 黑白名单
TITLE_WHITELIST = ["独舞", "solo", "练习室"]
TITLE_BLACKLIST = ["多人", "齐舞", "reaction"]
```

**作用：** 定义搜索的关键词库和过滤规则

#### 2️⃣ 配置类定义（PipelineConfig）

```python
@dataclass
class PipelineConfig:
    # 搜索参数
    max_pages_per_query: int = 5          # 默认值：5页
    search_order: str = "totalrank"       # 默认值：综合排序
    request_delay_sec: float = 1.5        # 默认值：1.5秒间隔
    
    # 过滤参数
    min_duration_sec: float = 30.0        # 默认值：最短30秒
    max_duration_sec: float = 3600.0      # 默认值：最长1小时
    
    # 评分参数
    solo_threshold: float = 0.75          # 默认值：0.75分
    audio_threshold: float = 0.60         # 默认值：0.60分
    
    # ... 60+ 个配置项
```

**作用：** 定义所有可配置参数的数据结构和默认值

#### 3️⃣ 默认配置实例

```python
DEFAULT_CONFIG = PipelineConfig()  # 使用所有默认值创建实例
```

**作用：** 提供一个开箱即用的默认配置对象

---

## 问题 3: cli.py 如何读取 config.py 的设定？

### 完整流程图解

```
┌─────────────────────────────────────────────────────────────┐
│ 第 0 步：导入默认配置                                         │
├─────────────────────────────────────────────────────────────┤
│ from .config import DEFAULT_CONFIG, PipelineConfig          │
│                                                             │
│ DEFAULT_CONFIG = {                                          │
│   max_pages_per_query: 5,                                   │
│   solo_threshold: 0.75,                                     │
│   audio_threshold: 0.60,                                    │
│   ...                                                       │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 1 步：用户运行命令                                         │
├─────────────────────────────────────────────────────────────┤
│ python -m bili_solo_pipeline.cli \                         │
│   --max_pages_per_query 10 \                               │
│   --solo_threshold 0.85 \                                  │
│   --out_dir data/test                                      │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 2 步：argparse 解析命令行参数                               │
├─────────────────────────────────────────────────────────────┤
│ parser = build_parser()                                     │
│ args = parser.parse_args()                                 │
│                                                             │
│ 结果：args = Namespace(                                      │
│   max_pages_per_query=10,      # 用户提供了                 │
│   solo_threshold=0.85,          # 用户提供了                │
│   out_dir='data/test',          # 用户提供了                │
│   audio_threshold=None,         # 用户没提供 -> None        │
│   min_duration_sec=None,        # 用户没提供 -> None        │
│   ...                                                       │
│ )                                                           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 3 步：从 DEFAULT_CONFIG 复制所有默认值                      │
├─────────────────────────────────────────────────────────────┤
│ cfg_fields = {                                              │
│   'max_pages_per_query': 5,        # 从 DEFAULT_CONFIG     │
│   'solo_threshold': 0.75,          # 从 DEFAULT_CONFIG     │
│   'audio_threshold': 0.60,         # 从 DEFAULT_CONFIG     │
│   'min_duration_sec': 30.0,        # 从 DEFAULT_CONFIG     │
│   ...                                                       │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 4 步：用命令行参数覆盖（仅覆盖非 None 的）                    │
├─────────────────────────────────────────────────────────────┤
│ override_map = {                                            │
│   'max_pages_per_query': 10,       # args 中有值           │
│   'solo_threshold': 0.85,          # args 中有值           │
│   'audio_threshold': None,         # args 中是 None        │
│   'min_duration_sec': None,        # args 中是 None        │
│ }                                                           │
│                                                             │
│ for key, value in override_map.items():                    │
│     if value is not None:          # 只覆盖有值的          │
│         cfg_fields[key] = value                            │
│                                                             │
│ 结果：cfg_fields = {                                         │
│   'max_pages_per_query': 10,       # ✅ 被覆盖了          │
│   'solo_threshold': 0.85,          # ✅ 被覆盖了          │
│   'audio_threshold': 0.60,         # ❌ 保持默认值        │
│   'min_duration_sec': 30.0,        # ❌ 保持默认值        │
│   ...                                                       │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 5 步（可选）：JSON 覆盖（优先级最高）                        │
├─────────────────────────────────────────────────────────────┤
│ 如果用户提供了 --config_override:                            │
│ --config_override '{"weight_solo": 0.6}'                   │
│                                                             │
│ overrides = json.loads('{"weight_solo": 0.6}')             │
│ cfg_fields['weight_solo'] = 0.6    # 覆盖任意字段          │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ 第 6 步：创建最终配置对象                                      │
├─────────────────────────────────────────────────────────────┤
│ config = PipelineConfig(**cfg_fields)                       │
│                                                             │
│ 最终配置：                                                    │
│   max_pages_per_query = 10         # 来自命令行             │
│   solo_threshold = 0.85            # 来自命令行             │
│   audio_threshold = 0.60           # 来自默认值             │
│   min_duration_sec = 30.0          # 来自默认值             │
│   weight_solo = 0.6                # 来自 JSON 覆盖        │
└─────────────────────────────────────────────────────────────┘
```

### 核心代码解析

```python
# cli.py 的 main() 函数

def main(argv=None):
    # 1. 解析命令行参数
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # 2. 从 DEFAULT_CONFIG 复制所有字段的默认值
    import dataclasses
    cfg_fields = {
        f.name: getattr(DEFAULT_CONFIG, f.name)
        for f in dataclasses.fields(DEFAULT_CONFIG)
    }
    # 结果：cfg_fields = {'max_pages_per_query': 5, 'solo_threshold': 0.75, ...}
    
    # 3. 构建覆盖映射（从 args 对象提取）
    override_map = {
        "max_pages_per_query": args.max_pages_per_query,  # 可能是 None
        "solo_threshold": args.solo_threshold,            # 可能是 None
        "audio_threshold": args.audio_threshold,          # 可能是 None
        # ... 所有命令行参数
    }
    
    # 4. 只覆盖非 None 的值
    for key, value in override_map.items():
        if value is not None:  # 关键：只覆盖用户实际提供的参数
            cfg_fields[key] = value
    
    # 5. (可选) JSON 覆盖
    if args.config_override:
        overrides = json.loads(args.config_override)
        for k, v in overrides.items():
            cfg_fields[k] = v
    
    # 6. 创建最终配置对象
    config = PipelineConfig(**cfg_fields)
    
    # 7. 传给 Pipeline 使用
    pipeline = Pipeline(config=config, ...)
    pipeline.run()
```

### 优先级总结

```
配置优先级（从低到高）：
┌────────────────────────────────────────────┐
│ 1️⃣ DEFAULT_CONFIG (config.py)             │  ← 最低优先级
│    max_pages_per_query = 5                │
├────────────────────────────────────────────┤
│ 2️⃣ 命令行参数 (--max_pages_per_query 10)  │  ← 中等优先级
│    覆盖 DEFAULT_CONFIG 中的值              │
├────────────────────────────────────────────┤
│ 3️⃣ JSON 覆盖 (--config_override '...')    │  ← 最高优先级
│    可以覆盖任意字段                        │
└────────────────────────────────────────────┘
```

---

## 完整示例：追踪一个参数的值

### 场景：用户想设置 solo_threshold

#### 方式 1：不提供任何参数（使用默认值）

```bash
python -m bili_solo_pipeline.cli --out_dir data/test
```

**值的来源：**
1. `config.py`: `solo_threshold: float = 0.75`  ← DEFAULT_CONFIG
2. `cli.py`: `args.solo_threshold = None`  ← 用户没提供
3. `cli.py`: `if None is not None: ...`  ← 不执行，保持默认值
4. **最终值：0.75**（来自 DEFAULT_CONFIG）

#### 方式 2：通过命令行参数

```bash
python -m bili_solo_pipeline.cli --solo_threshold 0.85 --out_dir data/test
```

**值的来源：**
1. `config.py`: `solo_threshold: float = 0.75`  ← DEFAULT_CONFIG
2. `cli.py`: `args.solo_threshold = 0.85`  ← 用户提供了
3. `cli.py`: `if 0.85 is not None: cfg_fields['solo_threshold'] = 0.85`  ← 覆盖
4. **最终值：0.85**（来自命令行）

#### 方式 3：通过 JSON 覆盖

```bash
python -m bili_solo_pipeline.cli \
  --solo_threshold 0.85 \
  --config_override '{"solo_threshold": 0.90}' \
  --out_dir data/test
```

**值的来源：**
1. `config.py`: `solo_threshold: float = 0.75`  ← DEFAULT_CONFIG
2. `cli.py`: `cfg_fields['solo_threshold'] = 0.85`  ← 命令行覆盖
3. `cli.py`: `cfg_fields['solo_threshold'] = 0.90`  ← JSON 再次覆盖
4. **最终值：0.90**（JSON 优先级最高）

---

## 设计优势

### 1. 默认值集中管理

```python
# ✅ 所有默认值都在 config.py 中定义
@dataclass
class PipelineConfig:
    solo_threshold: float = 0.75  # 清晰的默认值

# ❌ 如果默认值散落在代码各处
def some_function(threshold=0.75):  # 难以维护
    ...
```

### 2. 灵活的覆盖机制

```python
# 用户可以选择：
# - 什么都不提供 → 使用所有默认值
# - 只提供部分参数 → 只覆盖这些参数
# - 使用 JSON → 批量覆盖任意参数
```

### 3. 类型安全

```python
# dataclass 提供类型检查
config = PipelineConfig(
    solo_threshold=0.85,  # ✅ float
    solo_threshold="high",  # ❌ 类型错误
)
```

---

## 类比：餐厅点餐系统

```
DEFAULT_CONFIG = 默认套餐
  - 汉堡（beef）
  - 饮料（cola）
  - 薯条（medium）

命令行参数 = 客户定制
  --burger_type chicken     → 只改汉堡
  --drink_type juice        → 只改饮料
  （薯条保持默认 medium）

最终套餐：
  - 汉堡 = chicken  （来自客户）
  - 饮料 = juice    （来自客户）
  - 薯条 = medium   （保持默认）
```

---

## 总结

| 概念 | 作用 | 例子 |
|-----|------|------|
| **ArgumentParser** | 命令行参数解析器 | `parser = argparse.ArgumentParser()` |
| **ArgumentGroup** | 参数分组显示（仅用于帮助信息） | `search_grp = parser.add_argument_group("搜索参数")` |
| **config.py** | 定义配置结构和默认值 | `@dataclass class PipelineConfig: ...` |
| **DEFAULT_CONFIG** | 默认配置实例 | `DEFAULT_CONFIG = PipelineConfig()` |
| **cli.py** | 读取默认值 → 命令行覆盖 → JSON覆盖 | `config = PipelineConfig(**cfg_fields)` |

**一句话总结：**
- `config.py` 定义"能配什么"和"默认是什么"
- `cli.py` 定义"用户怎么改"（命令行参数）
- 运行时：默认值 → 命令行覆盖 → JSON 覆盖 → 最终配置
