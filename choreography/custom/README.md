# Bilibili 单人独舞视频候选集构建工具

> **仅用于研究候选集整理。请遵守 Bilibili 平台服务条款及相关版权法规。**
> 本工具不自动批量下载受版权保护视频，仅进行搜索、元数据抓取和质量评分。

---

## 目录

- [项目结构](#项目结构)  
- [快速开始](#快速开始)  
- [完整使用流程](#完整使用流程)  
- [配置参数说明](#配置参数说明)  
- [输出格式](#输出格式)  
- [调整阈值指南](#调整阈值指南)  
- [舞种词表说明](#舞种词表说明)  
- [常见问题](#常见问题)  
- [运行单元测试](#运行单元测试)

---

## 项目结构

```
custom/
├── bili_solo_pipeline/
│   ├── __init__.py
│   ├── models.py              # 数据结构定义（dataclass）
│   ├── config.py              # 全局配置 + 舞种词表
│   ├── utils.py               # 日志、速率限制、重试等工具
│   ├── search_client.py       # Bilibili 搜索客户端
│   ├── metadata_filter.py     # 元数据过滤（黑白名单 + 时长 + 舞种匹配）
│   ├── frame_sampler.py       # 视频抽帧
│   ├── person_detector.py     # 人体检测（YOLOv8）
│   ├── solo_scorer.py         # 单人独舞评分
│   ├── audio_quality_scorer.py # 音频质量评分
│   ├── dedup.py               # 去重（bvid + 标题 + 可选pHash）
│   ├── fusion_ranker.py       # 多分融合排序
│   ├── pipeline.py            # 主流程（串联所有模块）
│   └── cli.py                 # 命令行入口
├── tests/
│   ├── test_metadata_filter.py
│   ├── test_solo_scorer.py
│   ├── test_audio_scorer.py
│   └── test_dedup.py
├── data/                      # 输出目录（自动创建）
├── keywords.txt               # 自定义关键词示例
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
cd /network_space/server126/shared/sunyx/datasets/custom
pip install -r requirements.txt
```

**可选依赖：**

```bash
# 封面去重（pHash）
pip install imagehash Pillow

# 视频下载（使用前确认符合平台条款）
pip install yt-dlp
```

### 2. 仅搜索 + 元数据过滤（无需视频文件）

```bash
python -m bili_solo_pipeline.cli \
  --out_dir data/bili_solo_candidates \
  --max_pages_per_query 5 \
  --min_duration_sec 30
```

### 3. 带视觉/音频评分（需本地视频文件）

```bash
python -m bili_solo_pipeline.cli \
  --out_dir data/bili_solo_candidates \
  --video_dir /path/to/local/videos \
  --enable_audio_scoring true \
  --solo_threshold 0.75 \
  --audio_threshold 0.60
```

### 4. 使用自定义关键词文件

```bash
python -m bili_solo_pipeline.cli \
  --queries_file keywords.txt \
  --out_dir data/bili_solo_candidates
```

### 5. 断点续跑（从 dedup 阶段继续）

```bash
python -m bili_solo_pipeline.cli \
  --out_dir data/bili_solo_candidates \
  --run_from_stage dedup
```

---

## 完整使用流程

```
阶段 0: search   → 搜索关键词，收集原始元数据
阶段 1: filter   → 时长/黑白名单过滤，计算 metadata_score
阶段 2: dedup    → bvid/标题/pHash 去重
阶段 3: vision   → 抽帧 + YOLOv8 人体检测 + 单人评分（需本地视频）
阶段 4: audio    → ffmpeg 提取音轨 + 启发式音频质量评分（需本地视频）
阶段 5: rank     → 融合 solo/audio/metadata 三项分数排序
阶段 6: output   → 写出 all/keep/review/drop JSONL + summary.csv
```

缓存目录：`{out_dir}/.pipeline_cache/`  
每阶段完成后自动保存缓存，重启时自动加载。

---

## 配置参数说明

### 常用 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out_dir` | `data/bili_solo_candidates` | 输出目录 |
| `--queries_file` | 内置词表 | 自定义关键词文件 |
| `--max_pages_per_query` | 5 | 每个关键词最多搜几页 |
| `--min_duration_sec` | 30 | 最短时长（秒） |
| `--solo_threshold` | 0.75 | solo_score 达到此值 → keep |
| `--uncertain_threshold` | 0.45 | solo_score 在此值以上 → review |
| `--audio_threshold` | 0.60 | audio_score 达到此值 → clear_music |
| `--speech_ratio_max` | 0.35 | 语音占比超过此值 → speech_heavy |
| `--enable_audio_scoring` | true | 是否启用音频评分 |
| `--enable_pose` | false | 是否启用姿态估计 |
| `--video_dir` | 无 | 本地视频目录（触发视觉/音频检测） |
| `--run_from_stage` | search | 从哪个阶段开始运行 |
| `--request_delay_sec` | 1.5 | 每次请求间隔秒数（建议 >= 1.0） |
| `--use_cache` | true | 是否使用断点续跑缓存 |

### JSON 覆盖任意配置

```bash
python -m bili_solo_pipeline.cli \
  --out_dir data/out \
  --config_override '{"weight_solo": 0.6, "weight_audio": 0.25, "weight_metadata": 0.15}'
```

---

## 输出格式

### JSONL 字段（`all.jsonl` / `keep.jsonl` / `review.jsonl` / `drop.jsonl`）

```json
{
  "bvid": "BVxxxxxxxx",
  "title": "【独舞】jazz 练习室版 cover",
  "url": "https://www.bilibili.com/video/BVxxxxxxxx",
  "uploader": "UP主名",
  "duration_sec": 165.0,
  "publish_time": "2024-03-15T08:30:00Z",
  "tags": ["舞蹈", "jazz", "独舞"],
  "search_query_hit": "jazz 舞蹈 solo",
  "matched_genres": ["jazz"],
  "matched_keywords": ["jazz", "独舞"],
  "metadata_filter_pass": true,
  "metadata_filter_reasons": ["PASS:whitelist_hit(独舞, jazz)", "PASS:genre_hit(jazz)"],
  "metadata_score": 0.812,
  "vision_checked": true,
  "sampled_frames_count": 55,
  "person_stats": {
    "valid_person_frame_ratio": 0.94,
    "single_person_frame_ratio": 0.89,
    "crowded_frame_ratio": 0.05,
    "avg_max_bbox_area_ratio": 0.18
  },
  "solo_score": 0.831,
  "solo_label": "likely_solo",
  "solo_reasons": ["PASS:single_person_ratio=0.89", "PASS:solo_score=0.831 >= threshold=0.75"],
  "audio_checked": true,
  "audio_stats": {
    "audio_present_ratio": 0.96,
    "speech_ratio": 0.08,
    "music_confidence": 0.79,
    "snr_proxy": 22.5,
    "clipping_ratio": 0.001,
    "loudness_mean": -18.3,
    "loudness_stability": 0.87,
    "beat_strength": 0.65
  },
  "audio_score": 0.74,
  "audio_label": "clear_music",
  "audio_reasons": ["PASS:speech_ratio=0.08", "audio_score=0.740 label=clear_music"],
  "final_score": 0.782,
  "final_recommendation": "keep"
}
```

### summary.csv

按搜索关键词和舞种统计命中数、过滤数、最终保留数：

```csv
search_query,total,metadata_passed,keep,review,drop
jazz 舞蹈 solo,80,52,18,24,10
独舞,120,75,28,31,16
...

# Genre Coverage
genre,total_matched,keep,review
jazz,95,21,30
hiphop,88,19,28
...
```

---

## 调整阈值指南

### 想要更保守（减少误报）
```bash
--solo_threshold 0.85      # 提高单人确认门槛
--audio_threshold 0.70     # 提高音频质量要求
--speech_ratio_max 0.25    # 降低允许的语音占比
```

### 想要更宽松（增加召回）
```bash
--solo_threshold 0.65      # 降低 keep 门槛（更多 keep）
--uncertain_threshold 0.30 # 降低 review 门槛（减少 drop）
--audio_threshold 0.50     # 降低音频要求
```

### 调整融合权重
```bash
# 更重视音频质量
--config_override '{"weight_solo":0.40,"weight_audio":0.45,"weight_metadata":0.15}'

# 纯依赖元数据（无视觉/音频检测时）
--config_override '{"weight_solo":0.0,"weight_audio":0.0,"weight_metadata":1.0}'
```

---

## 舞种词表说明

内置舞种映射见 `bili_solo_pipeline/config.py`，**特别注意**：

| 内部标签 | 实际含义 | 说明 |
|---------|---------|------|
| `hoping` | popping/机械舞 | 内部模型标签名为 hoping，实为 popping/poppin（震感舞）|
| `tai` | 傣族舞 + 泰国舞 | 两者风格相似，词表涵盖两类，需人工细分 |
| `korea` | K-POP 风格 + 朝鲜族民间舞 | 两类差异大，统一召回后人工区分 |
| `shenyun` | 中国古典舞"身韵"技法 | 注意区分"身韵"（技法）与"神韵艺术团"|

如需修改词表，直接编辑 `config.py` 中的 `GENRE_KEYWORD_MAP`。

---

## 常见问题

**Q: 搜索返回错误 -412（风控触发）？**  
A: 增加请求间隔：`--request_delay_sec 3.0`，或提供 Cookie 文件：`--cookies_file cookies.json`

**Q: 没有本地视频，如何只做搜索+元数据过滤？**  
A: 不提供 `--video_dir`，pipeline 自动跳过视觉/音频检测，仅基于 `metadata_score` 排序。

**Q: 视觉评分全是 0 / vision_checked=False？**  
A: 确认 `--video_dir` 目录下有 `{bvid}.mp4` 格式的文件（如 `BVxxxxxxxx.mp4`）。

**Q: 如何只更新音频评分而不重跑搜索？**  
A: `--run_from_stage audio --use_cache true`

**Q: Cookie 文件格式？**  
A: JSON 字典格式 `{"SESSDATA": "xxx", "bili_jct": "xxx"}`，或从浏览器扩展导出的 Netscape 格式 JSON 数组。

---

## 运行单元测试

```bash
cd /network_space/server126/shared/sunyx/datasets/custom
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=bili_solo_pipeline --cov-report=term-missing
```

---

## 版权与合规声明

- 本工具仅用于**研究候选集整理**，输出结果供人工复核
- 不自动批量下载视频；如需本地视频请自行确认版权及平台条款
- `audio_score` 高仅代表"音乐质量好"，**不代表版权可用性**
- 最终数据使用前请进行人工复核和合规确认
