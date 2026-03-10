"""
fusion_ranker.py
================
融合多项分数，输出最终推荐结果。

融合策略（可配置权重）：
  solo_score    权重最高（视觉检测直接反映单人独舞属性）
  audio_score   权重次高（音频质量对训练数据友好性至关重要）
  metadata_score 权重最低（辅助参考）

特殊规则（可通过 config 开关）：
  1. speech_heavy 视频：若 solo_score 高但 audio_label=speech_heavy，
     进入 review（因为可能是口令舞/教学视频，需人工确认）
  2. 多机位节目救援：视觉不稳定（uncertain/solo_score 中等）但 audio_score 高
     进入 review 而不是 drop（避免错杀节目类精彩独舞）
  3. 完全未检测（vision_checked=False 或 audio_checked=False）：
     降级到 review 区（不直接 drop，避免遗漏）

标签定义：
  keep   : final_score >= final_keep_threshold 且通过所有硬规则
  review : 分数处于中间区间或触发特殊规则，需人工复核
  drop   : final_score < drop_threshold 或命中严重拒绝条件
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models import VideoRecord
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# 最终 keep/drop 阈值（基于 final_score）
_FINAL_KEEP_THRESHOLD = 0.65
_FINAL_DROP_THRESHOLD = 0.30


@dataclass
class FusionResult:
    final_score: float
    final_recommendation: str  # keep / review / drop
    reasons: list[str]


class FusionRanker:
    """多分数融合排序器。

    Args:
        config: 全局配置
        keep_threshold: final_score >= 此值 -> keep（默认 0.65）
        drop_threshold: final_score < 此值 -> drop（默认 0.30）
    """

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        keep_threshold: float = _FINAL_KEEP_THRESHOLD,
        drop_threshold: float = _FINAL_DROP_THRESHOLD,
    ) -> None:
        self._cfg = config
        self._keep_threshold = keep_threshold
        self._drop_threshold = drop_threshold

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def rank(self, rec: VideoRecord) -> FusionResult:
        """对单个视频计算最终融合分数和推荐标签。

        Args:
            rec: 已填充视觉+音频评分的 VideoRecord

        Returns:
            FusionResult
        """
        reasons: list[str] = []
        cfg = self._cfg

        # ---- 权重归一化 ----
        w_solo = cfg.weight_solo
        w_audio = cfg.weight_audio
        w_meta = cfg.weight_metadata
        total_w = w_solo + w_audio + w_meta

        # 如未做视觉检测，solo_score 默认 0.5（中性）
        solo_score = rec.solo_score if rec.vision_checked else 0.5
        audio_score = rec.audio_score if rec.audio_checked else 0.5
        meta_score = rec.metadata_score

        if not rec.vision_checked:
            reasons.append("WARN:vision_not_checked—using_neutral_solo_score")
        if not rec.audio_checked:
            reasons.append("WARN:audio_not_checked—using_neutral_audio_score")

        # ---- weighted sum ----
        raw_score = (
            solo_score * w_solo +
            audio_score * w_audio +
            meta_score * w_meta
        ) / total_w
        final_score = round(float(raw_score), 4)

        # ---- 标签文本评分加成 ----
        # metadata_filter_pass=False 直接 drop
        if not rec.metadata_filter_pass:
            reasons.append("REJECT:metadata_filter_failed")
            return FusionResult(
                final_score=final_score,
                final_recommendation="drop",
                reasons=reasons,
            )

        # ---- 特殊规则 ----
        recommendation = self._apply_special_rules(
            rec, final_score, reasons, solo_score, audio_score
        )

        reasons.append(
            f"final_score={final_score:.3f} "
            f"(solo={solo_score:.3f}×{w_solo}, "
            f"audio={audio_score:.3f}×{w_audio}, "
            f"meta={meta_score:.3f}×{w_meta})"
        )

        logger.debug(
            "FusionRanker: bvid=%s final=%.3f rec=%s",
            rec.bvid,
            final_score,
            recommendation,
        )
        return FusionResult(
            final_score=final_score,
            final_recommendation=recommendation,
            reasons=reasons,
        )

    def rank_and_update(self, rec: VideoRecord) -> VideoRecord:
        """就地更新 VideoRecord 的 final_score 和 final_recommendation，返回更新后的记录。"""
        result = self.rank(rec)
        rec.final_score = result.final_score
        rec.final_recommendation = result.final_recommendation
        return rec

    def rank_batch(self, records: list[VideoRecord]) -> list[VideoRecord]:
        """批量融合排序，返回按 final_score 降序排列的列表。"""
        for rec in records:
            self.rank_and_update(rec)
        records.sort(key=lambda r: r.final_score, reverse=True)
        logger.info(
            "FusionRanker batch: keep=%d review=%d drop=%d",
            sum(1 for r in records if r.final_recommendation == "keep"),
            sum(1 for r in records if r.final_recommendation == "review"),
            sum(1 for r in records if r.final_recommendation == "drop"),
        )
        return records

    # ------------------------------------------------------------------
    # 内部规则
    # ------------------------------------------------------------------

    def _apply_special_rules(
        self,
        rec: VideoRecord,
        final_score: float,
        reasons: list[str],
        solo_score: float,
        audio_score: float,
    ) -> str:
        cfg = self._cfg

        # 规则 1：强制 drop - solo_label 明确多人 且 分数很低
        if rec.solo_label == "likely_multi" and final_score < self._drop_threshold:
            reasons.append("RULE:likely_multi+low_score -> drop")
            return "drop"

        # 规则 2：无人画面且音频也差 -> drop
        if rec.solo_label == "no_person" and audio_score < 0.3:
            reasons.append("RULE:no_person+poor_audio -> drop")
            return "drop"

        # 规则 3：语音过多救援逻辑
        # 如果 audio_label=speech_heavy 且 solo_score 高 -> review
        if rec.audio_label == "speech_heavy" and solo_score > cfg.solo_threshold:
            reasons.append(
                "RULE:speech_heavy_but_good_solo -> review "
                "(may_be_instructional_dance_or_counted_ballet)"
            )
            return "review"

        # 规则 4：多机位节目救援
        # 视觉不确定 但 音频好 -> review 而不是 drop
        if (
            cfg.unstable_vision_audio_rescue
            and rec.solo_label in ("uncertain",)
            and audio_score >= cfg.audio_threshold
            and final_score >= self._drop_threshold
        ):
            reasons.append(
                "RULE:uncertain_vision_but_good_audio -> review "
                "(possible_multi-camera_program)"
            )
            return "review"

        # 规则 5：视觉未检测但元数据好 -> review
        if not rec.vision_checked and rec.metadata_score > 0.6:
            reasons.append("RULE:vision_unchecked+good_metadata -> review")
            return "review"

        # ---- 普通阈值判断 ----
        if final_score >= self._keep_threshold:
            # 额外检查：如果 speech_ratio 过高，降级到 review
            if (
                rec.audio_checked
                and rec.audio_stats
                and rec.audio_stats.speech_ratio > cfg.speech_ratio_max
            ):
                reasons.append(
                    f"RULE:good_score_but_high_speech({rec.audio_stats.speech_ratio:.2f}) -> review"
                )
                return "review"
            return "keep"
        elif final_score >= self._drop_threshold:
            return "review"
        else:
            return "drop"
