"""
metadata_filter.py
==================
第一轮元数据过滤。

过滤逻辑（可解释，每条规则独立输出 reason）：
  1. 时长精确判断（>= min_duration_sec，注意搜索时长桶是粗粒度的）
  2. 时长上限（不要超长合集）
  3. 标题/标签黑名单命中 -> 拒绝
  4. 标题/标签白名单命中 -> 加分
  5. 舞种关键词匹配 -> 填写 matched_genres / matched_keywords
  6. 音频相关加分/降权词
  7. 综合计算 metadata_score（0~1）

注意：
  - "只靠标题无法准确识别单人独舞"，因此此阶段仅做粗筛
  - metadata_score 较低的视频仍然可通过（pass=True），但后续 fusion_ranker 会降权
  - 若黑名单命中 -> metadata_filter_pass=False，直接跳过视觉/音频检测节省资源
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import (
    PipelineConfig,
    DEFAULT_CONFIG,
    GENRE_KEYWORD_MAP,
    TITLE_WHITELIST,
    TITLE_BLACKLIST,
    TITLE_SOFT_PENALTY,
    AUDIO_BOOST_KEYWORDS,
    AUDIO_PENALTY_KEYWORDS,
)
from .models import RawSearchItem, VideoRecord
from .search_client import raw_item_to_video_url
from .utils import contains_any, normalize_title

import datetime

logger = logging.getLogger(__name__)


class MetadataFilter:
    """对 RawSearchItem 做元数据过滤，输出 VideoRecord（含过滤结果）。

    Args:
        config: 全局配置
        extra_whitelist: 附加白名单词（用户自定义）
        extra_blacklist: 附加黑名单词
    """

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        extra_whitelist: Optional[list[str]] = None,
        extra_blacklist: Optional[list[str]] = None,
    ) -> None:
        self._cfg = config

        # 合并舞种关键词到白名单（所有舞种词本身就是白名单词）
        genre_kws_flat = [kw for kws in GENRE_KEYWORD_MAP.values() for kw in kws]
        self._whitelist = list(set(TITLE_WHITELIST + genre_kws_flat + (extra_whitelist or [])))
        self._blacklist = list(set(TITLE_BLACKLIST + (extra_blacklist or [])))
        self._soft_penalty = list(TITLE_SOFT_PENALTY)
        logger.debug(
            "MetadataFilter initialized: %d whitelist / %d blacklist / %d soft_penalty keywords",
            len(self._whitelist),
            len(self._blacklist),
            len(self._soft_penalty),
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def filter(self, item: RawSearchItem) -> VideoRecord:
        """过滤单条搜索结果，返回填充好第一阶段字段的 VideoRecord。"""
        rec = self._raw_to_record(item)
        reasons: list[str] = []
        score_parts: list[float] = []

        # ---- 1. 时长精确判断：严格限制在 30s - 5min ----
        if rec.duration_sec < 30:
            reasons.append(
                f"REJECT:duration_too_short({rec.duration_sec:.0f}s < 30s)"
            )
            rec.metadata_filter_pass = False
            rec.metadata_filter_reasons = reasons
            rec.metadata_score = 0.0
            return rec

        if rec.duration_sec > 300:  # 5 minutes
            reasons.append(
                f"REJECT:duration_too_long({rec.duration_sec:.0f}s > 300s)"
            )
            rec.metadata_filter_pass = False
            rec.metadata_filter_reasons = reasons
            rec.metadata_score = 0.0
            return rec

        # ---- 2. 播放量下限 ----
        if rec.view_count < self._cfg.min_view_count:
            reasons.append(f"REJECT:view_count_too_low({rec.view_count} < {self._cfg.min_view_count})")
            rec.metadata_filter_pass = False
            rec.metadata_filter_reasons = reasons
            rec.metadata_score = 0.0
            return rec

        # ---- 3. 标题 + 标签黑名单 ----
        combined_text = rec.title + " " + " ".join(rec.tags)
        blacklist_hits = contains_any(combined_text, self._blacklist)
        if blacklist_hits:
            # 黑名单命中 -> 直接拒绝
            reasons.append(f"REJECT:blacklist_hit({', '.join(blacklist_hits[:5])})")
            rec.metadata_filter_pass = False
            rec.metadata_filter_reasons = reasons
            rec.metadata_score = 0.0
            return rec

        # ---- 4. 标题白名单命中 ----
        whitelist_hits = contains_any(combined_text, self._whitelist)
        if whitelist_hits:
            reasons.append(f"PASS:whitelist_hit({', '.join(whitelist_hits[:5])})")
            white_score = min(1.0, len(whitelist_hits) / 3.0)
            score_parts.append(("whitelist", white_score, 0.40))
        else:
            reasons.append("WARN:no_whitelist_keyword_hit")
            score_parts.append(("whitelist", 0.1, 0.40))

        # ---- 5. 舞种关键词匹配 ----
        matched_genres, matched_kws = self._match_genres(combined_text)
        rec.matched_genres = matched_genres
        rec.matched_keywords = matched_kws
        if matched_genres:
            reasons.append(f"PASS:genre_hit({', '.join(matched_genres[:5])})")
            genre_score = min(1.0, len(matched_genres) * 0.5)
            score_parts.append(("genre", genre_score, 0.35))
        else:
            reasons.append("INFO:no_genre_matched")
            score_parts.append(("genre", 0.0, 0.35))

        # ---- 6. 音频相关关键词 ----
        audio_boost = contains_any(combined_text, AUDIO_BOOST_KEYWORDS)
        audio_penalty = contains_any(combined_text, AUDIO_PENALTY_KEYWORDS)
        
        # 如果命中 audio_penalty，直接拒绝
        if audio_penalty:
            reasons.append(f"REJECT:audio_penalty_hit({', '.join(audio_penalty[:3])})")
            rec.metadata_filter_pass = False
            rec.metadata_filter_reasons = reasons
            rec.metadata_score = 0.0
            return rec
        
        if audio_boost:
            reasons.append(f"AUDIO_BOOST:{', '.join(audio_boost[:3])}")
            audio_meta_score = min(1.0, 0.7 + 0.3 * len(audio_boost))
            score_parts.append(("audio_meta", audio_meta_score, 0.25))
        else:
            score_parts.append(("audio_meta", 0.5, 0.25))

        # ---- 计算综合 metadata_score ----
        if isinstance(score_parts[0], tuple):
            total_weight = sum(w for _, _, w in score_parts)
            weighted_sum = sum(s * w for _, s, w in score_parts)
        else:
            total_weight = 1.0
            weighted_sum = float(score_parts[0])

        rec.metadata_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
        rec.metadata_filter_pass = True
        reasons.append(f"metadata_score={rec.metadata_score:.3f}")
        rec.metadata_filter_reasons = reasons

        logger.debug(
            "MetadataFilter: bvid=%s pass=%s score=%.3f reasons=%s",
            rec.bvid,
            rec.metadata_filter_pass,
            rec.metadata_score,
            reasons,
        )
        return rec

    def filter_batch(self, items: list[RawSearchItem]) -> tuple[list[VideoRecord], list[VideoRecord]]:
        """批量过滤，返回 (passed_list, rejected_list)。"""
        passed, rejected = [], []
        for item in items:
            rec = self.filter(item)
            if rec.metadata_filter_pass:
                passed.append(rec)
            else:
                rejected.append(rec)
        logger.info(
            "MetadataFilter batch: %d passed / %d rejected (total=%d)",
            len(passed),
            len(rejected),
            len(items),
        )
        return passed, rejected

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_to_record(item: RawSearchItem) -> VideoRecord:
        """将 RawSearchItem 转换为 VideoRecord（填写基础字段）。"""
        publish_time = ""
        if item.publish_ts > 0:
            try:
                publish_time = datetime.datetime.fromtimestamp(
                    item.publish_ts, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (OSError, ValueError):
                publish_time = str(item.publish_ts)

        url = raw_item_to_video_url(item.bvid)

        return VideoRecord(
            bvid=item.bvid,
            aid=item.aid,
            title=item.title,
            url=url,
            uploader=item.uploader,
            duration_sec=item.duration_sec,
            publish_time=publish_time,
            view_count=item.view_count,
            tags=item.tags,
            tid=item.tid,
            tid_name=item.tid_name,
            cover_url=item.cover_url,
            search_query_hit=item.search_query,
        )

    @staticmethod
    def _match_genres(text: str) -> tuple[list[str], list[str]]:
        """在文本中匹配所有舞种关键词，返回 (matched_genres, matched_keywords)。"""
        text_low = text.lower()
        matched_genres: list[str] = []
        matched_keywords: list[str] = []
        for genre, kws in GENRE_KEYWORD_MAP.items():
            hits = [kw for kw in kws if kw.lower() in text_low]
            if hits:
                matched_genres.append(genre)
                matched_keywords.extend(hits)
        return matched_genres, list(set(matched_keywords))


