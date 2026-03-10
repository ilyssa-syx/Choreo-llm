"""
dedup.py
========
去重模块，支持三级去重策略：
  1. bvid 精确去重（必须）
  2. 标题归一化去重（基于编辑距离或 token 重叠）
  3. 封面图感知哈希（pHash）去重（可选，需 imagehash + PIL）

输出重复映射关系以供追溯。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from .models import VideoRecord, DedupResult
from .config import PipelineConfig, DEFAULT_CONFIG
from .utils import normalize_title

logger = logging.getLogger(__name__)

try:
    import imagehash  # type: ignore
    from PIL import Image  # type: ignore
    import io, requests as _req
    _PHASH_AVAILABLE = True
except ImportError:
    _PHASH_AVAILABLE = False
    logger.info(
        "imagehash/Pillow not installed; cover pHash dedup disabled. "
        "Install with: pip install imagehash Pillow"
    )


class Deduplicator:
    """多级去重器。

    Args:
        config: 全局配置
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        self._enable_phash = config.enable_phash_dedup and _PHASH_AVAILABLE
        if config.enable_phash_dedup and not _PHASH_AVAILABLE:
            logger.warning(
                "enable_phash_dedup=True but imagehash/Pillow not installed. "
                "pHash dedup will be skipped."
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def dedup(self, records: list[VideoRecord]) -> DedupResult:
        """对视频记录列表进行去重。

        去重优先级：bvid > title > pHash （后者覆盖前者）

        Args:
            records: 候选视频列表（顺序按搜索时间/分数，前者优先保留）

        Returns:
            DedupResult，包含去重后列表和重复映射
        """
        result = DedupResult()
        seen_bvids: dict[str, str] = {}          # bvid -> canonical_bvid
        seen_titles: dict[str, str] = {}         # normalized_title -> canonical_bvid
        seen_phashes: dict[str, str] = {}        # phash_str -> canonical_bvid
        unique: list[VideoRecord] = []
        dup_map: dict[str, str] = {}

        for rec in records:
            dup_bvid = self._check_bvid_dup(rec.bvid, seen_bvids)
            if dup_bvid:
                dup_map[rec.bvid] = dup_bvid
                logger.debug("DUP(bvid): %s -> %s", rec.bvid, dup_bvid)
                continue

            norm_title = normalize_title(rec.title)
            dup_title = self._check_title_dup(
                rec.bvid, norm_title, seen_titles, self._cfg.title_sim_threshold
            )
            if dup_title:
                dup_map[rec.bvid] = dup_title
                logger.debug("DUP(title): %s -> %s  title='%s'", rec.bvid, dup_title, rec.title)
                continue

            if self._enable_phash and rec.cover_url:
                dup_phash = self._check_phash_dup(rec.bvid, rec.cover_url, seen_phashes)
                if dup_phash:
                    dup_map[rec.bvid] = dup_phash
                    logger.debug("DUP(phash): %s -> %s", rec.bvid, dup_phash)
                    continue

            # 不是重复，加入 seen
            seen_bvids[rec.bvid] = rec.bvid
            seen_titles[norm_title] = rec.bvid
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

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _check_bvid_dup(bvid: str, seen: dict[str, str]) -> Optional[str]:
        """精确 bvid 去重。"""
        return seen.get(bvid)

    @staticmethod
    def _check_title_dup(
        bvid: str,
        norm_title: str,
        seen: dict[str, str],
        threshold: float,
    ) -> Optional[str]:
        """基于 token-jaccard 相似度的标题去重。

        对于短标题（<5 字符），直接精确匹配。
        """
        if len(norm_title) < 5:
            return seen.get(norm_title)

        for seen_title, canonical_bvid in seen.items():
            sim = _jaccard_similarity(norm_title, seen_title)
            if sim >= threshold:
                return canonical_bvid
        return None

    @staticmethod
    def _check_phash_dup(
        bvid: str,
        cover_url: str,
        seen: dict[str, str],
        max_distance: int = 8,
    ) -> Optional[str]:
        """封面图感知哈希去重（pHash）。

        pHash 汉明距离 <= max_distance 视为重复（同视频重传）。
        """
        if not _PHASH_AVAILABLE:
            return None
        try:
            resp = _req.get(cover_url, timeout=5)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            phash = imagehash.phash(img)
        except Exception as exc:
            logger.debug("pHash fetch/compute failed for bvid=%s: %s", bvid, exc)
            return None

        phash_str = str(phash)
        for seen_phash_str, canonical_bvid in seen.items():
            try:
                seen_hash = imagehash.hex_to_hash(seen_phash_str)
                distance = phash - seen_hash
                if distance <= max_distance:
                    return canonical_bvid
            except Exception:
                continue

        seen[phash_str] = bvid
        return None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """将文本拆分为字符 N-gram + 词（宽松分词）。"""
    # 英文单词分割 + 中文字符级
    tokens = set()
    # 英文词
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens.update(words)
    # 中文字符 2-gram
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i + 1]
        if "\u4e00" <= c1 <= "\u9fff" and "\u4e00" <= c2 <= "\u9fff":
            tokens.add(c1 + c2)
    # 单个中文字符
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            tokens.add(c)
    return tokens


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard 相似度（基于字符/词 token）。"""
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union
