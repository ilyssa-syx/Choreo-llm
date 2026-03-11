# bili_solo_pipeline 架构说明

## 问题 1：`python -m bili_solo_pipeline.cli` 是如何工作的？

### Python 模块运行方式解释

`python -m bili_solo_pipeline.cli` 这个命令中：
- `-m` 是 Python 的参数，表示"作为模块运行"
- `bili_solo_pipeline.cli` 是模块路径，不是文件路径
- 实际运行的文件是 `bili_solo_pipeline/cli.py`

**工作原理：**
1. Python 会把 `bili_solo_pipeline` 识别为一个包（因为有 `__init__.py`）
2. `cli` 是这个包下的一个模块（对应 `cli.py` 文件）
3. 运行时会执行 `cli.py` 文件中的代码
4. 因为 `cli.py` 末尾有 `if __name__ == "__main__": main()`，所以会调用 `main()` 函数

**等价写法：**
```bash
# 方式 1（推荐）：作为模块运行
python -m bili_solo_pipeline.cli

# 方式 2：直接运行文件（不推荐，可能有导入问题）
python bili_solo_pipeline/cli.py

# 方式 3：通过包导入再运行
python -c "from bili_solo_pipeline.cli import main; main()"
```

## 问题 2：文件之间的调用关系

### 调用层次图

```
用户命令行
    ↓
cli.py (命令行入口)
    ├── 解析参数 (argparse)
    ├── 构建配置 (config.py → PipelineConfig)
    └── 创建并运行 Pipeline (pipeline.py → Pipeline.run())
            ↓
pipeline.py (主流程编排)
    ├── Stage 0: search (使用 search_client.py)
    │       └── BiliSearchClient.search_many_queries()
    │
    ├── Stage 1: filter (使用 metadata_filter.py)
    │       └── MetadataFilter.filter_batch()
    │
    ├── Stage 2: dedup (使用 dedup.py)
    │       └── Deduplicator.dedup()
    │
    ├── Stage 3: vision (如果提供了 video_dir)
    │       ├── frame_sampler.py → FrameSampler.sample_frames()
    │       ├── person_detector.py → PersonDetector.detect_batch()
    │       └── solo_scorer.py → SoloScorer.score()
    │
    ├── Stage 4: audio (如果 enable_audio_scoring=true)
    │       └── audio_quality_scorer.py → AudioQualityScorer.score()
    │
    ├── Stage 5: rank (融合排序)
    │       └── fusion_ranker.py → FusionRanker.rank_batch()
    │
    ├── Stage 6: output (输出结果)
    │       └── 写入 JSONL 和 CSV
    │
    └── Stage 7: download (如果 enable_download=true)
            └── downloader.py → VideoDownloader.download_batch()
```

### 详细调用流程

#### 1. 启动阶段 (cli.py)

```python
# cli.py 的 main() 函数
def main():
    # 1. 解析命令行参数
    parser = build_parser()
    args = parser.parse_args()
    
    # 2. 构建配置对象
    config = PipelineConfig(...)
    
    # 3. 创建 Pipeline 实例
    pipeline = Pipeline(
        config=config,
        cookies_file=args.cookies_file,
        run_from_stage=args.run_from_stage,
        video_dir=args.video_dir,
    )
    
    # 4. 运行 Pipeline
    records = pipeline.run()
```

#### 2. Pipeline 初始化 (pipeline.py)

```python
# pipeline.py 的 __init__() 方法
class Pipeline:
    def __init__(self, config, ...):
        # 创建所有需要的子模块实例
        self._search_client = BiliSearchClient(config)
        self._metadata_filter = MetadataFilter(config)
        self._deduplicator = Deduplicator(config)
        self._frame_sampler = FrameSampler(...)
        self._person_detector = PersonDetector(...)
        self._solo_scorer = SoloScorer(config)
        self._audio_scorer = AudioQualityScorer(config)
        self._fusion_ranker = FusionRanker(config)
```

#### 3. Pipeline 运行流程 (pipeline.py)

```python
def run(self) -> list[VideoRecord]:
    # Stage 0: 搜索
    raw_items = self._stage_search()
    #   ↓ 调用 search_client.search_many_queries()
    
    # Stage 1: 过滤
    passed, rejected = self._stage_filter(raw_items)
    #   ↓ 调用 metadata_filter.filter_batch()
    
    # Stage 2: 去重
    unique = self._deduplicator.dedup(passed)
    #   ↓ 调用 dedup.py 的去重逻辑
    
    # Stage 3: 视觉检测（如果有视频）
    if video_dir:
        unique = self._stage_vision(unique)
        #   ↓ frame_sampler.sample_frames()
        #   ↓ person_detector.detect_batch()
        #   ↓ solo_scorer.score()
    
    # Stage 4: 音频检测（如果启用）
    if enable_audio_scoring:
        unique = self._stage_audio(unique)
        #   ↓ audio_quality_scorer.score()
    
    # Stage 5: 融合排序
    unique = self._fusion_ranker.rank_batch(unique)
    
    # Stage 6: 输出结果
    self._stage_output(unique)
    
    # Stage 7: 下载（可选）
    if enable_download:
        self._stage_download(unique)
    
    return unique
```

### 核心文件职责说明

| 文件 | 职责 | 被谁调用 |
|------|------|---------|
| `cli.py` | 命令行入口，参数解析 | 用户通过 `python -m` 调用 |
| `pipeline.py` | 主流程编排，串联所有阶段 | cli.py 的 main() 函数 |
| `config.py` | 全局配置、舞种词表 | 所有模块读取配置 |
| `models.py` | 数据结构定义 (dataclass) | 所有模块使用数据结构 |
| `utils.py` | 工具函数（日志、重试等） | 所有模块使用工具函数 |
| `search_client.py` | B站搜索 API 封装 | pipeline._stage_search() |
| `metadata_filter.py` | 元数据过滤（时长、标签等） | pipeline._stage_filter() |
| `dedup.py` | 去重（bvid/标题/pHash） | pipeline.run() Stage 2 |
| `frame_sampler.py` | 视频抽帧 | pipeline._stage_vision() |
| `person_detector.py` | YOLOv8 人体检测 | pipeline._stage_vision() |
| `solo_scorer.py` | 计算单人独舞分数 | pipeline._stage_vision() |
| `audio_quality_scorer.py` | 音频质量评分 | pipeline._stage_audio() |
| `fusion_ranker.py` | 融合多个分数排序 | pipeline.run() Stage 5 |
| `downloader.py` | 视频下载（yt-dlp） | pipeline._stage_download() |

### 数据流向

```
命令行参数
    ↓
PipelineConfig (config.py)
    ↓
搜索关键词
    ↓
RawSearchItem[] (models.py)
    ↓
VideoRecord[] (经过过滤)
    ↓
VideoRecord[] (去重后)
    ↓
VideoRecord[] (添加 vision 评分)
    ↓
VideoRecord[] (添加 audio 评分)
    ↓
VideoRecord[] (添加 final_score)
    ↓
输出: all.jsonl, keep.jsonl, review.jsonl, drop.jsonl, summary.csv
```

### 断点续跑机制

Pipeline 在每个阶段完成后会保存缓存：
```
{out_dir}/.pipeline_cache/
    ├── stage_search.jsonl          # 原始搜索结果
    ├── stage_filter_passed.jsonl   # 通过过滤的记录
    ├── stage_filter_rejected.jsonl # 被拒绝的记录
    ├── stage_dedup_unique.jsonl    # 去重后的记录
    ├── stage_vision_scored.jsonl   # 视觉评分后
    ├── stage_audio_scored.jsonl    # 音频评分后
    └── stage_ranked.jsonl          # 最终排序后
```

使用 `--run_from_stage <stage>` 可以从某个阶段继续运行，前面的阶段直接从缓存加载。

### 关键设计模式

1. **命令模式**: cli.py 将用户命令转换为 Pipeline 操作
2. **管道模式**: Pipeline 将数据依次通过各个处理阶段
3. **策略模式**: 各个 Scorer/Filter 实现各自的评分/过滤策略
4. **工厂模式**: Pipeline 根据配置创建所需的子模块实例

## 示例：完整运行追踪

当你运行：
```bash
python -m bili_solo_pipeline.cli --out_dir data/test --max_pages_per_query 2
```

实际执行流程：
1. Python 加载 `bili_solo_pipeline/cli.py` 模块
2. 执行 `cli.py` 中的 `if __name__ == "__main__": main()`
3. `main()` 解析参数，创建 `PipelineConfig` 对象
4. 创建 `Pipeline` 实例，初始化所有子模块
5. 调用 `pipeline.run()`：
   - 调用 `search_client.search_many_queries()` 搜索视频
   - 调用 `metadata_filter.filter_batch()` 过滤
   - 调用 `deduplicator.dedup()` 去重
   - （如果有视频）调用视觉检测链
   - （如果启用）调用音频检测
   - 调用 `fusion_ranker.rank_batch()` 排序
   - 输出结果文件
6. 程序结束

## 总结

- **cli.py** 是入口，负责参数解析和 Pipeline 启动
- **pipeline.py** 是核心，负责编排所有处理阶段
- 其他模块都是被 pipeline.py 调用的工具模块
- 数据在各个阶段之间流动，逐步添加评分和标签
- 通过缓存机制支持断点续跑
