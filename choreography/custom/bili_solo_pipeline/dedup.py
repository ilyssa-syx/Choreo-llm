"""
dedup.py
========
去重模块，基于 bvid 精确去重。
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import VideoRecord, DedupResult
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class Deduplicator:
    """基于 bvid 的去重器。

    Args:
        config: 全局配置
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def dedup(self, records: list[VideoRecord]) -> DedupResult:
        """对视频记录列表进行去重（仅基于 bvid）。

        Args:
            records: 候选视频列表（顺序按搜索时间/分数，前者优先保留）

        Returns:
            DedupResult，包含去重后列表和重复映射
        """
        result = DedupResult()
        seen_bvids: set[str] = set()
        unique: list[VideoRecord] = []
        dup_map: dict[str, str] = {}

        for rec in records:
            if rec.bvid in seen_bvids:
                dup_map[rec.bvid] = rec.bvid
                logger.debug("DUP(bvid): %s", rec.bvid)
                continue

            # 不是重复，加入 seen
            seen_bvids.add(rec.bvid)
            unique.append(rec)

        result.unique_records = unique
        result.duplicate_map = dup_map
        result.removed_count = len(records) - len(unique)
        logger.info(
            "Dedup: %d input -> %d unique (removed %d duplicates)",
            len(records),
            len(unique),
            result.removed_count,
        )
        return result

