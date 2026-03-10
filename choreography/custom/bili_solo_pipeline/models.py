"""
models.py
=========
所有模块共享的 dataclass 数据结构定义。

设计原则：
- 每个阶段填充自己负责的字段，其余字段保持 None
- 最终序列化到 JSONL 时只输出非 None 字段（或按需序列化全部）
- 时间戳统一使用 ISO-8601 字符串，方便 JSON 序列化
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ---------------------------------------------------------------------------
# 1. 单视频完整记录
# ---------------------------------------------------------------------------

@dataclass
class PersonStats:
    """视觉检测统计（由 solo_scorer.py 汇总）"""
    valid_person_frame_ratio: float = 0.0     # 至少检测到 1 人的帧占比
    single_person_frame_ratio: float = 0.0   # 恰好 1 人的帧占比（核心指标）
    crowded_frame_ratio: float = 0.0          # >=2 人帧占比
    avg_max_bbox_area_ratio: float = 0.0      # 最大人物 bbox 面积均值（画面占比）
    motion_score: Optional[float] = None      # 运动估计（可选）


@dataclass
class AudioStats:
    """音频质量统计（由 audio_quality_scorer.py 汇总）"""
    audio_present_ratio: float = 0.0      # 非静音段占比
    speech_ratio: float = 0.0             # 语音占比（过高 -> 口播/解说）
    music_confidence: float = 0.0         # 音乐置信度（0~1）
    snr_proxy: float = 0.0                # 简易信噪比代理（dB，越高越好）
    clipping_ratio: float = 0.0           # 削波/失真帧占比
    loudness_mean: float = -60.0          # 平均响度（LUFS 近似）
    loudness_stability: float = 0.0       # 响度稳定性（1 - normalized_std）
    beat_strength: Optional[float] = None # 节拍强度（可选，供后续排序）


@dataclass
class VideoRecord:
    """单个视频候选的完整记录，贯穿整个 pipeline"""

    # ---- 基础元数据（search_client 填充）----
    bvid: str = ""
    aid: Optional[str] = None
    title: str = ""
    url: str = ""
    uploader: str = ""
    duration_sec: float = 0.0
    publish_time: str = ""
    view_count: int = 0
    tags: list[str] = field(default_factory=list)
    tid: Optional[int] = None              # B 站分区 ID
    tid_name: Optional[str] = None        # 分区名称
    cover_url: Optional[str] = None       # 封面图 URL

    # ---- 搜索命中信息（search_client / metadata_filter 填充）----
    search_query_hit: str = ""            # 触发命中的搜索关键词
    matched_genres: list[str] = field(default_factory=list)      # 命中舞种列表
    matched_keywords: list[str] = field(default_factory=list)    # 命中的具体关键词

    # ---- 第一轮元数据过滤（metadata_filter 填充）----
    metadata_filter_pass: bool = False
    metadata_filter_reasons: list[str] = field(default_factory=list)
    metadata_score: float = 0.0           # 0~1

    # ---- 视觉检测结果（frame_sampler + person_detector + solo_scorer 填充）----
    vision_checked: bool = False
    sampled_frames_count: int = 0
    person_stats: Optional[PersonStats] = None
    solo_score: float = 0.0               # 0~1
    solo_label: str = ""                  # likely_solo / uncertain / likely_multi / no_person
    solo_reasons: list[str] = field(default_factory=list)

    # ---- 音频质量（audio_quality_scorer 填充）----
    audio_checked: bool = False
    audio_stats: Optional[AudioStats] = None
    audio_score: float = 0.0              # 0~1
    audio_label: str = ""                 # clear_music / speech_heavy / noisy_audio / low_volume / uncertain_audio
    audio_reasons: list[str] = field(default_factory=list)

    # ---- 融合排序（fusion_ranker 填充）----
    final_score: float = 0.0              # 0~1
    final_recommendation: str = "drop"   # keep / review / drop

    # ---- 下载（downloader 填充）----
    downloaded_path: Optional[str] = None  # 本地 .mp4 文件路径

    # ---- 内部字段（不导出到最终 JSONL）----
    _local_video_path: Optional[str] = field(default=None, repr=False)
    _local_audio_path: Optional[str] = field(default=None, repr=False)
    _cover_phash: Optional[str] = field(default=None, repr=False)

    # ----------------------------------------------------------------
    def to_dict(self, include_internal: bool = False) -> dict:
        """转为普通字典，嵌套 dataclass 也展开"""
        d = asdict(self)
        if not include_internal:
            for k in list(d.keys()):
                if k.startswith("_"):
                    del d[k]
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoRecord":
        """从字典重建（用于断点续跑加载缓存）"""
        d = dict(d)
        if "person_stats" in d and d["person_stats"] is not None:
            d["person_stats"] = PersonStats(**d["person_stats"])
        if "audio_stats" in d and d["audio_stats"] is not None:
            d["audio_stats"] = AudioStats(**d["audio_stats"])
        # 去掉内部字段，避免 dataclass 报错
        internal = [k for k in d if k.startswith("_")]
        for k in internal:
            del d[k]
        return cls(**d)


# ---------------------------------------------------------------------------
# 2. 搜索原始结果（search_client 内部使用）
# ---------------------------------------------------------------------------

@dataclass
class RawSearchItem:
    """Bilibili 搜索接口返回的单条原始结果"""
    bvid: str
    aid: str
    title: str
    uploader: str
    duration_str: str        # 原始格式如 "3:25"
    duration_sec: float      # 解析后秒数
    publish_ts: int          # Unix timestamp
    view_count: int
    cover_url: str
    tid: int
    tid_name: str
    tags: list[str] = field(default_factory=list)
    search_query: str = ""


# ---------------------------------------------------------------------------
# 3. 去重结果
# ---------------------------------------------------------------------------

@dataclass
class DedupResult:
    unique_records: list[VideoRecord] = field(default_factory=list)
    duplicate_map: dict[str, str] = field(default_factory=dict)  # bvid -> canonical_bvid
    removed_count: int = 0
