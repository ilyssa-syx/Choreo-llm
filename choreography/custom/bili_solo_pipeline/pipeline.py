"""
pipeline.py
===========
主流程：串联所有模块，支持断点续跑和阶段跳过。

流程：
  多关键词搜索
    -> 元数据过滤（第一轮）
    -> 全局去重（bvid + 标题 + 可选pHash）
    -> [可选] 抽帧 + 人体检测 + 单人打分
    -> [可选] 音频质量评分
    -> 融合排序
    -> 输出 JSONL / CSV

断点续跑（use_cache=True）：
  - 每个阶段完成后将结果写入 {cache_dir}/stage_{name}.jsonl
  - 重启时若缓存存在则跳过已完成的阶段
  - 通过 --run_from_stage 参数指定从哪个阶段开始重跑

阶段名称（有序）：
  search -> filter -> dedup -> vision -> audio -> rank -> output
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm  # type: ignore

from .config import (
    PipelineConfig,
    DEFAULT_CONFIG,
    GENRE_KEYWORD_MAP,
    GENERAL_DANCE_KEYWORDS,
)
from .models import VideoRecord, RawSearchItem
from .search_client import BiliSearchClient
from .metadata_filter import MetadataFilter
from .dedup import Deduplicator
from .frame_sampler import FrameSampler
from .person_detector import PersonDetector
from .solo_scorer import SoloScorer
from .audio_quality_scorer import AudioQualityScorer
from .fusion_ranker import FusionRanker
from .downloader import VideoDownloader
from .utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# 阶段常量（download 移到 vision 之前，在 dedup 后先下载视频）
STAGES = ["search", "filter", "dedup", "download", "vision", "audio", "rank", "output", "cleanup"]


class Pipeline:
    """主 Pipeline，串联所有处理阶段。

    Args:
        config: 全局配置
        queries: 搜索关键词列表；None 则从 config 自动生成
        cookies_file: B 站 Cookie 文件路径（可选，提升搜索成功率）
        run_from_stage: 从哪个阶段开始运行（前面阶段从缓存加载）
        video_dir: 本地视频目录（供视觉/音频模块使用）
    """

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        queries: Optional[list[str]] = None,
        cookies_file: Optional[str] = None,
        run_from_stage: str = "search",
        video_dir: Optional[str] = None,
    ) -> None:
        self._cfg = config
        self._queries = queries or self._build_queries()
        self._cookies = self._load_cookies(cookies_file)
        self._run_from = run_from_stage
        self._video_dir = Path(video_dir) if video_dir else None

        # 输出目录
        self._out_dir = ensure_dir(config.out_dir)
        self._cache_dir = ensure_dir(
            os.path.join(config.out_dir, config.cache_dir)
        ) if config.use_cache else None

        # 子模块（延迟初始化耗资源模块）
        self._search_client = BiliSearchClient(config, cookies=self._cookies)
        self._metadata_filter = MetadataFilter(config)
        self._deduplicator = Deduplicator(config)
        self._frame_sampler = FrameSampler(
            sample_every_sec=config.sample_every_sec,
            max_frames=config.max_frames,
            skip_head_sec=config.skip_head_sec,
        )
        self._person_detector: Optional[PersonDetector] = None  # 延迟加载
        self._solo_scorer = SoloScorer(config)
        self._audio_scorer = AudioQualityScorer(config)
        self._fusion_ranker = FusionRanker(config)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> list[VideoRecord]:
        """运行完整 pipeline，返回最终排序列表。"""
        logger.info("=== Bilibili Solo Dance Pipeline START ===")
        logger.info("Keywords: %d queries", len(self._queries))
        logger.info("Output dir: %s", self._out_dir)
        logger.info("Run from stage: %s", self._run_from)

        stage_idx = STAGES.index(self._run_from) if self._run_from in STAGES else 0

        # ---- Stage 0: Search ----
        if stage_idx <= STAGES.index("search"):
            raw_items = self._stage_search()
            self._save_cache("search", raw_items)
        else:
            raw_items = self._load_cache_raw("search")

        # ---- Stage 1: Filter ----
        if stage_idx <= STAGES.index("filter"):
            passed, rejected = self._stage_filter(raw_items)
            all_records = passed + rejected
            self._save_cache_records("filter_passed", passed)
            self._save_cache_records("filter_rejected", rejected)
        else:
            passed = self._load_cache_records("filter_passed")
            rejected = self._load_cache_records("filter_rejected")
            all_records = passed + rejected

        logger.info("After filter: %d passed / %d rejected", len(passed), len(rejected))

        # ---- Stage 2: Dedup ----
        if stage_idx <= STAGES.index("dedup"):
            dedup_result = self._deduplicator.dedup(passed)
            unique = dedup_result.unique_records
            self._save_cache_records("dedup_unique", unique)
        else:
            unique = self._load_cache_records("dedup_unique")

        logger.info("After dedup: %d unique records", len(unique))

        # ---- Stage 3: Download (before vision, so we have videos to analyze) ----
        if self._cfg.enable_download and stage_idx <= STAGES.index("download"):
            # 先下载所有候选视频（download_review=True 确保全部下载）
            self._stage_download_candidates(unique)

        # 无论是否启用下载，只要 video_dir 未显式指定，就尝试从 download_dir 推断
        # 优先: enable_download 时直接使用 download_dir
        # 兜底: download_dir/candidates/ 目录存在时也自动使用（支持 --run_from_stage vision 跳过下载阶段）
        if self._video_dir is None:
            download_dir = Path(self._cfg.download_dir)
            if self._cfg.enable_download:
                self._video_dir = download_dir
                logger.info("Set video_dir to download_dir: %s", self._video_dir)
            elif (download_dir / "candidates").exists():
                self._video_dir = download_dir
                logger.info("Auto-detected video_dir from %s/candidates/: %s", download_dir, self._video_dir)

        # ---- Stage 4: Vision ----
        if stage_idx <= STAGES.index("vision"):
            unique = self._stage_vision(unique)
            self._save_cache_records("vision_scored", unique)
        else:
            unique = self._load_cache_records("vision_scored")

        # ---- Stage 5: Audio ----
        if self._cfg.enable_audio_scoring:
            if stage_idx <= STAGES.index("audio"):
                unique = self._stage_audio(unique)
                self._save_cache_records("audio_scored", unique)
            else:
                unique = self._load_cache_records("audio_scored")

        # ---- Stage 6: Rank ----
        if stage_idx <= STAGES.index("rank"):
            unique = self._fusion_ranker.rank_batch(unique)
            self._save_cache_records("ranked", unique)
        else:
            unique = self._load_cache_records("ranked")

        # ---- Stage 7: Output ----
        self._stage_output(unique)

        # ---- Stage 8: Cleanup (delete drop videos) ----
        if self._cfg.enable_download and stage_idx <= STAGES.index("cleanup"):
            self._cleanup_drop_videos(unique)

        logger.info("=== Pipeline DONE. Total candidates: %d ===", len(unique))
        return unique

    # ------------------------------------------------------------------
    # 各阶段实现
    # ------------------------------------------------------------------

    def _stage_search(self) -> list[RawSearchItem]:
        """搜索阶段：收集所有关键词的原始搜索结果。

        若 config.target_total_hours > 0：
          - 自动将单关键词页数扩展到 max_pages_when_targeting（默认 50）
          - 按 target * search_oversample_rate 的原始时长作为停止条件
            （oversample 补偿后续过滤的损耗，默认 4× 即搜 40h 原始素材来得到 10h 成品）
        """
        logger.info("Stage: search (%d queries)", len(self._queries))
        seen_bvids: set[str] = set()
        items: list[RawSearchItem] = []
        accumulated_sec: float = 0.0

        target_sec: float = 0.0
        effective_max_pages = self._cfg.max_pages_per_query
        if self._cfg.target_total_hours > 0:
            target_sec = (
                self._cfg.target_total_hours
                * self._cfg.search_oversample_rate
                * 3600.0
            )
            effective_max_pages = max(
                self._cfg.max_pages_per_query,
                self._cfg.max_pages_when_targeting,
            )
            logger.info(
                "目标 %.1f 小时 × oversample %.1f× → 搜索原始上限 %.1f 小时，"
                "单关键词最多 %d 页",
                self._cfg.target_total_hours,
                self._cfg.search_oversample_rate,
                target_sec / 3600,
                effective_max_pages,
            )

        with tqdm(total=len(self._queries), desc="Searching", unit="query") as pbar:
            for query, item in self._search_client.search_many_queries(
                self._queries,
                max_pages=effective_max_pages,
            ):
                if item.bvid not in seen_bvids:
                    seen_bvids.add(item.bvid)
                    items.append(item)
                    accumulated_sec += item.duration_sec
                    if target_sec > 0 and accumulated_sec >= target_sec:
                        logger.info(
                            "原始时长上限已达到 %.1f / %.1f 小时，停止搜索。",
                            accumulated_sec / 3600,
                            target_sec / 3600,
                        )
                        break
                pbar.set_postfix(
                    query=query[:20],
                    total=len(items),
                    hours=f"{accumulated_sec/3600:.1f}h",
                )
            pbar.n = pbar.total
            pbar.refresh()

        logger.info(
            "Search total: %d unique raw items, accumulated %.2f hours",
            len(items),
            accumulated_sec / 3600,
        )
        return items

    def _stage_filter(
        self, items: list[RawSearchItem]
    ) -> tuple[list[VideoRecord], list[VideoRecord]]:
        """元数据过滤阶段。"""
        logger.info("Stage: filter (%d items)", len(items))
        passed, rejected = [], []
        for item in tqdm(items, desc="Filtering", unit="video"):
            rec = self._metadata_filter.filter(item)
            if rec.metadata_filter_pass:
                passed.append(rec)
            else:
                rejected.append(rec)
        passed_hours = sum(r.duration_sec for r in passed) / 3600
        logger.info(
            "Filter done: passed=%d (%.2f h) rejected=%d (pass_rate=%.1f%%)",
            len(passed), passed_hours, len(rejected),
            100.0 * len(passed) / max(len(items), 1),
        )
        if self._cfg.target_total_hours > 0 and passed_hours < self._cfg.target_total_hours:
            logger.warning(
                "通过过滤的视频仅 %.2f h，未达 %.1f h 目标。"
                "可尝试: 增大 --search_oversample_rate（当前 %.1f）或 --max_pages_when_targeting。",
                passed_hours, self._cfg.target_total_hours, self._cfg.search_oversample_rate,
            )
        return passed, rejected

    def _stage_vision(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """视觉检测阶段：抽帧 -> 人体检测 -> 单人打分。"""
        logger.info("Stage: vision (%d records)", len(records))

        if self._person_detector is None:
            self._person_detector = PersonDetector(
                model_path=self._cfg.yolo_model,
                conf_threshold=self._cfg.yolo_conf_threshold,
                enable_pose=self._cfg.enable_pose,
            )

        for rec in tqdm(records, desc="Vision scoring", unit="video"):
            # 如果已经做过视觉检测（断点续跑），跳过
            if rec.vision_checked:
                continue

            video_path = self._find_local_video(rec.bvid)
            if video_path is None:
                logger.debug("No local video for bvid=%s, skipping vision.", rec.bvid)
                continue

            try:
                timestamps, frames = self._frame_sampler.sample_from_file(video_path)
                if not frames:
                    logger.warning("No frames sampled for bvid=%s", rec.bvid)
                    continue

                detections = self._person_detector.detect_frames(timestamps, frames)
                score_result = self._solo_scorer.score(detections)

                rec.vision_checked = True
                rec.sampled_frames_count = len(frames)
                rec.person_stats = score_result.person_stats
                rec.solo_score = score_result.score
                rec.solo_label = score_result.label
                rec.solo_reasons = score_result.reasons

                del frames  # 释放内存
            except Exception as exc:
                logger.error("Vision scoring failed for bvid=%s: %s", rec.bvid, exc)

        return records

    def _stage_audio(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """音频质量评分阶段。"""
        logger.info("Stage: audio (%d records)", len(records))

        for rec in tqdm(records, desc="Audio scoring", unit="video"):
            if rec.audio_checked:
                continue

            video_path = self._find_local_video(rec.bvid)
            if video_path is None:
                logger.debug("No local video for bvid=%s, skipping audio.", rec.bvid)
                continue

            try:
                result = self._audio_scorer.score_from_video(video_path, bvid=rec.bvid)
                rec.audio_checked = True
                rec.audio_stats = result.audio_stats
                rec.audio_score = result.audio_score
                rec.audio_label = result.audio_label
                rec.audio_reasons = result.audio_reasons
            except Exception as exc:
                logger.error("Audio scoring failed for bvid=%s: %s", rec.bvid, exc)

        return records

    def _stage_download_candidates(self, records: list[VideoRecord]) -> None:
        """下载候选视频阶段：在 vision 检测前下载所有候选视频。"""
        logger.info("Stage: download_candidates (%d videos)", len(records))
        
        downloader = VideoDownloader(
            config=self._cfg,
            cookies_file=self._cfg.download_cookies_file or None,
        )
        
        try:
            # force_all_candidates=True：跳过 label/score 过滤，
            # 下载所有候选视频（因为还没做 vision/audio 检测）
            downloader.download_batch(records, force_all_candidates=True)
        except RuntimeError as exc:
            logger.error("Download failed: %s", exc)

    def _cleanup_drop_videos(self, records: list[VideoRecord]) -> None:
        """清理阶段：
        1. 将 keep/review 的视频从 candidates/ 移动到对应目录
        2. 删除 drop 的视频文件，节省存储空间
        """
        if self._video_dir is None:
            logger.warning("No video_dir set, skip cleanup")
            return
        
        candidates_dir = self._video_dir / "candidates"
        if not candidates_dir.exists():
            logger.info("No candidates directory, skip cleanup")
            return
        
        # 先移动 keep 和 review 的文件
        keep_review_records = [r for r in records if r.final_recommendation in ("keep", "review")]
        moved_count = 0
        for rec in keep_review_records:
            bvid = rec.bvid
            for ext in ["mp4", "flv", "webm", "mkv", "avi"]:
                src_path = candidates_dir / f"{bvid}.{ext}"
                if src_path.exists():
                    dest_dir = self._video_dir / rec.final_recommendation
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / src_path.name
                    try:
                        src_path.rename(dest_path)
                        rec.downloaded_path = str(dest_path)
                        moved_count += 1
                        logger.debug("Moved: %s -> %s", src_path.name, rec.final_recommendation)
                    except Exception as exc:
                        logger.warning("Failed to move %s: %s", src_path, exc)
        
        logger.info("Moved %d videos to keep/review directories", moved_count)
        
        # 删除 drop 的文件（包括 candidates 目录中剩余的）
        drop_records = [r for r in records if r.final_recommendation == "drop"]
        deleted_count = 0
        deleted_size = 0
        
        for rec in drop_records:
            bvid = rec.bvid
            # 尝试从 candidates 和 drop 目录删除
            for dir_name in ["candidates", "drop"]:
                check_dir = self._video_dir / dir_name
                if not check_dir.exists():
                    continue
                for ext in ["mp4", "flv", "webm", "mkv", "avi"]:
                    video_path = check_dir / f"{bvid}.{ext}"
                    if video_path.exists():
                        try:
                            file_size = video_path.stat().st_size
                            video_path.unlink()
                            deleted_count += 1
                            deleted_size += file_size
                            logger.debug("Deleted: %s (%.2f MB)", video_path.name, file_size / 1024 / 1024)
                        except Exception as exc:
                            logger.warning("Failed to delete %s: %s", video_path, exc)
        
        logger.info(
            "Cleanup done: moved %d files, deleted %d files, freed %.2f GB",
            moved_count,
            deleted_count,
            deleted_size / 1024 / 1024 / 1024,
        )

    def _stage_output(self, records: list[VideoRecord]) -> None:
        """输出阶段：写各类 JSONL 和 summary CSV。

        采用追加合并模式：先读取已有 all.jsonl 中的历史记录，
        以 bvid 为主键去重后与本轮新记录合并，再写出全量文件。
        这样多次运行不会清空历史数据，且自动去除重复视频。
        """
        logger.info("Stage: output (%d records this run)", len(records))
        cfg = self._cfg
        out = self._out_dir

        # ---- 加载历史记录（如有） ----
        existing_path = out / "all.jsonl"
        historical: list[VideoRecord] = []
        if existing_path.exists():
            historical = self._load_jsonl(existing_path)
            logger.info("Loaded %d historical records from %s", len(historical), existing_path)

        # ---- 合并去重：以 bvid 为 key，新记录优先覆盖旧记录 ----
        merged: dict[str, VideoRecord] = {r.bvid: r for r in historical}
        new_count = 0
        for r in records:
            if r.bvid not in merged:
                new_count += 1
            merged[r.bvid] = r  # 新记录覆盖（保留最新评分）
        all_records = list(merged.values())
        logger.info(
            "Merged: %d historical + %d new (deduped) = %d total",
            len(historical),
            new_count,
            len(all_records),
        )

        keep = [r for r in all_records if r.final_recommendation == "keep"]
        review = [r for r in all_records if r.final_recommendation == "review"]
        drop = [r for r in all_records if r.final_recommendation == "drop"]

        # 全量
        self._write_jsonl(all_records, out / "all.jsonl")
        if cfg.write_keep:
            self._write_jsonl(keep, out / "keep.jsonl")
        if cfg.write_review:
            self._write_jsonl(review, out / "review.jsonl")
        if cfg.write_drop:
            self._write_jsonl(drop, out / "drop.jsonl")

        if cfg.write_summary_csv:
            self._write_summary_csv(all_records, out / "summary.csv")

        self._write_duration_summary(
            all_records=all_records,
            keep=keep,
            review=review,
            drop=drop,
            path=out / "duration_summary.csv",
        )

        logger.info(
            "Output: keep=%d review=%d drop=%d total=%d (this run added %d new)",
            len(keep),
            len(review),
            len(drop),
            len(all_records),
            new_count,
        )

    # ------------------------------------------------------------------
    # 缓存工具
    # ------------------------------------------------------------------

    def _save_cache(self, name: str, items: list[RawSearchItem]) -> None:
        if not self._cache_dir:
            return
        path = self._cache_dir / f"stage_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
        logger.debug("Cache saved: %s (%d items)", path, len(items))

    def _load_cache_raw(self, name: str) -> list[RawSearchItem]:
        if not self._cache_dir:
            return []
        path = self._cache_dir / f"stage_{name}.jsonl"
        if not path.exists():
            logger.warning("Cache not found: %s", path)
            return []
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    from .models import RawSearchItem
                    data = json.loads(line)
                    # 过滤不认识的字段
                    fields = RawSearchItem.__dataclass_fields__.keys()
                    data = {k: v for k, v in data.items() if k in fields}
                    items.append(RawSearchItem(**data))
        logger.info("Cache loaded: %s (%d items)", path, len(items))
        return items

    def _save_cache_records(self, name: str, records: list[VideoRecord]) -> None:
        if not self._cache_dir:
            return
        path = self._cache_dir / f"stage_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_json() + "\n")
        logger.debug("Cache saved: %s (%d records)", path, len(records))

    def _load_cache_records(self, name: str) -> list[VideoRecord]:
        if not self._cache_dir:
            return []
        path = self._cache_dir / f"stage_{name}.jsonl"
        if not path.exists():
            logger.warning("Cache not found: %s", path)
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(VideoRecord.from_dict(json.loads(line)))
                    except Exception as exc:
                        logger.warning("Failed to load record from cache: %s", exc)
        logger.info("Cache loaded: %s (%d records)", path, len(records))
        return records

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    def _find_local_video(self, bvid: str) -> Optional[Path]:
        """在 video_dir 中查找与 bvid 对应的视频文件。
        
        查找顺序：candidates/ -> keep/ -> review/ -> drop/
        """
        if self._video_dir is None:
            return None
        
        # 优先在 candidates 目录查找（候选阶段下载的位置）
        for subdir in ["candidates", "keep", "review", "drop", ""]:
            search_dir = self._video_dir / subdir if subdir else self._video_dir
            if not search_dir.exists():
                continue
            for ext in ["mp4", "flv", "webm", "mkv", "avi"]:
                candidate = search_dir / f"{bvid}.{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _build_queries(self) -> list[str]:
        """从配置构建搜索关键词列表。"""
        queries: list[str] = []
        if self._cfg.use_genre_keywords:
            for genre, kws in GENRE_KEYWORD_MAP.items():
                queries.extend(kws)
        if self._cfg.use_general_keywords:
            queries.extend(GENERAL_DANCE_KEYWORDS)
        # 如果有外部关键词文件，读取
        if self._cfg.queries_file:
            try:
                with open(self._cfg.queries_file, encoding="utf-8") as f:
                    user_queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                queries = user_queries  # 用户文件覆盖内置查询
                logger.info("Loaded %d queries from %s", len(queries), self._cfg.queries_file)
            except Exception as exc:
                logger.warning("Failed to load queries_file: %s", exc)
        # 去重保持顺序
        seen: set[str] = set()
        result: list[str] = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                result.append(q)
        return result

    @staticmethod
    def _load_cookies(cookies_file: Optional[str]) -> Optional[dict[str, str]]:
        """从 JSON 文件加载 Cookie。"""
        if not cookies_file:
            return None
        try:
            with open(cookies_file, encoding="utf-8") as f:
                cookies = json.load(f)
            if isinstance(cookies, dict):
                return cookies
            # Netscape 格式 list
            if isinstance(cookies, list):
                return {c["name"]: c["value"] for c in cookies if "name" in c and "value" in c}
        except Exception as exc:
            logger.warning("Failed to load cookies file '%s': %s", cookies_file, exc)
        return None

    @staticmethod
    def _write_jsonl(records: list[VideoRecord], path: Path) -> None:
        """将记录列表写入 JSONL 文件（覆盖写，调用方负责去重合并）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.to_json() + "\n")
        logger.info("Written %d records -> %s", len(records), path)

    @staticmethod
    def _load_jsonl(path: Path) -> list[VideoRecord]:
        """从 JSONL 文件加载 VideoRecord 列表，跳过解析失败的行。"""
        records: list[VideoRecord] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(VideoRecord.from_dict(json.loads(line)))
                    except Exception as exc:
                        logger.warning("Skipping malformed line in %s: %s", path, exc)
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
        return records

    @staticmethod
    def _write_summary_csv(records: list[VideoRecord], path: Path) -> None:
        """写汇总 CSV：各关键词命中数、过滤数、最终保留数、舞种覆盖率。"""
        path.parent.mkdir(parents=True, exist_ok=True)

        # 按 search_query_hit 统计
        query_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "passed": 0, "keep": 0, "review": 0, "drop": 0}
        )
        genre_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "keep": 0, "review": 0}
        )
        for rec in records:
            q = rec.search_query_hit or "unknown"
            query_stats[q]["total"] += 1
            if rec.metadata_filter_pass:
                query_stats[q]["passed"] += 1
            query_stats[q][rec.final_recommendation] += 1

            for genre in rec.matched_genres:
                genre_stats[genre]["total"] += 1
                if rec.final_recommendation in ("keep", "review"):
                    genre_stats[genre][rec.final_recommendation] += 1

        rows = []
        for q, s in query_stats.items():
            rows.append({
                "search_query": q,
                "total": s["total"],
                "metadata_passed": s["passed"],
                "keep": s["keep"],
                "review": s["review"],
                "drop": s["drop"],
            })

        # 舞种覆盖摘要追加
        genre_rows = []
        for genre, s in genre_stats.items():
            genre_rows.append({
                "genre": genre,
                "total_matched": s["total"],
                "keep": s["keep"],
                "review": s["review"],
            })

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            f.write("\n# Genre Coverage\n")
            if genre_rows:
                writer = csv.DictWriter(f, fieldnames=list(genre_rows[0].keys()))
                writer.writeheader()
                writer.writerows(genre_rows)

        logger.info("Summary CSV written -> %s", path)

    @staticmethod
    def _write_duration_summary(
        all_records: list[VideoRecord],
        keep: list[VideoRecord],
        review: list[VideoRecord],
        drop: list[VideoRecord],
        path: Path,
    ) -> None:
        """写时长汇总 CSV：总量与各推荐桶的总时长/平均时长。"""
        path.parent.mkdir(parents=True, exist_ok=True)

        def _row(split: str, recs: list[VideoRecord]) -> dict[str, float | int | str]:
            total_sec = sum(max(0.0, r.duration_sec) for r in recs)
            count = len(recs)
            avg_sec = (total_sec / count) if count else 0.0
            return {
                "split": split,
                "count": count,
                "total_duration_sec": round(total_sec, 2),
                "total_duration_hours": round(total_sec / 3600.0, 4),
                "avg_duration_sec": round(avg_sec, 2),
                "avg_duration_min": round(avg_sec / 60.0, 2),
            }

        rows = [
            _row("all", all_records),
            _row("keep", keep),
            _row("review", review),
            _row("drop", drop),
        ]

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "count",
                    "total_duration_sec",
                    "total_duration_hours",
                    "avg_duration_sec",
                    "avg_duration_min",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        logger.info(
            "Duration summary written -> %s (all=%.2f h, keep=%.2f h, review=%.2f h, drop=%.2f h)",
            path,
            rows[0]["total_duration_hours"],
            rows[1]["total_duration_hours"],
            rows[2]["total_duration_hours"],
            rows[3]["total_duration_hours"],
        )
