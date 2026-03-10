"""
solo_scorer.py
==============
根据逐帧检测结果对视频进行"单人独舞"打分。

评分规则（全部可配置）：
  valid_person_frame_ratio  : 至少 1 人的帧占比（避免空镜头）
  single_person_frame_ratio : 恰好 1 人的帧占比（核心指标）
  dominant_person_ratio     : 主体人物面积是否稳定且足够大（避免远景多人）
  crowded_frame_ratio       : >=2 人帧占比（过高直接拒绝）
  motion_score (optional)   : 通过 bbox 中心/面积变化粗略衡量运动（避免静态封面）

输出标签：
  likely_solo   : score >= solo_threshold
  uncertain     : uncertain_threshold <= score < solo_threshold
  likely_multi  : crowded_frame_ratio 过高
  no_person     : valid_person_frame_ratio 极低

设计说明：
  - 节目类多机位切换会导致误判（帧间人数/位置剧烈变化），
    此类情况 solo_score 可能偏低，pipeline 应结合 audio_score 判断
    进入 review 而不是直接 drop
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .person_detector import FrameDetection
from .models import PersonStats
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SoloScoreResult:
    score: float = 0.0
    label: str = "uncertain"
    reasons: list[str] = field(default_factory=list)
    person_stats: Optional[PersonStats] = None


class SoloScorer:
    """单人独舞评分器。

    Args:
        config: 全局配置
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def score(self, detections: list[FrameDetection]) -> SoloScoreResult:
        """根据帧检测列表计算单人独舞评分。

        Args:
            detections: FrameDetection 列表（来自 person_detector）

        Returns:
            SoloScoreResult
        """
        result = SoloScoreResult()
        reasons: list[str] = []

        if not detections:
            result.label = "no_person"
            result.score = 0.0
            reasons.append("REJECT:no_frames_detected")
            result.reasons = reasons
            return result

        n_total = len(detections)

        # ---- 基础统计 ----
        n_valid = sum(1 for d in detections if d.person_count >= 1)
        n_single = sum(1 for d in detections if d.person_count == 1)
        n_crowded = sum(1 for d in detections if d.person_count >= 2)
        max_area_ratios = [d.max_bbox_area_ratio for d in detections if d.person_count >= 1]

        valid_ratio = n_valid / n_total
        single_ratio = n_single / n_total
        crowded_ratio = n_crowded / n_total
        avg_max_area = float(np.mean(max_area_ratios)) if max_area_ratios else 0.0

        stats = PersonStats(
            valid_person_frame_ratio=round(valid_ratio, 4),
            single_person_frame_ratio=round(single_ratio, 4),
            crowded_frame_ratio=round(crowded_ratio, 4),
            avg_max_bbox_area_ratio=round(avg_max_area, 4),
        )

        # ---- 运动估计（可选）----
        motion_score = self._calc_motion_score(detections)
        if motion_score is not None:
            stats.motion_score = round(motion_score, 4)

        result.person_stats = stats

        # ---- 规则判断 ----

        # 规则 1：几乎没有人 -> no_person
        if valid_ratio < self._cfg.min_valid_person_ratio:
            result.label = "no_person"
            result.score = 0.05
            reasons.append(
                f"REJECT:low_valid_person_ratio({valid_ratio:.2f} < {self._cfg.min_valid_person_ratio:.2f})"
            )
            result.reasons = reasons
            return result

        # 规则 2：多人帧太多 -> likely_multi
        if crowded_ratio > self._cfg.max_crowded_ratio:
            result.label = "likely_multi"
            result.score = max(0.0, 0.4 - (crowded_ratio - self._cfg.max_crowded_ratio) * 1.5)
            reasons.append(
                f"REJECT:high_crowded_ratio({crowded_ratio:.2f} > {self._cfg.max_crowded_ratio:.2f})"
            )
            result.reasons = reasons
            return result

        # ---- 计算 solo_score ----
        score_components: list[tuple[str, float, float]] = []

        # 单人帧比例（权重最高）
        score_components.append(("single_ratio", single_ratio, 0.45))
        if single_ratio < self._cfg.min_single_person_ratio:
            reasons.append(
                f"WARN:low_single_person_ratio({single_ratio:.2f} < {self._cfg.min_single_person_ratio:.2f})"
            )
        else:
            reasons.append(f"PASS:single_person_ratio={single_ratio:.2f}")

        # 有效人体帧比例
        score_components.append(("valid_ratio", valid_ratio, 0.20))

        # 主体面积大小（越大说明人物越清晰）
        # min_dominant_area=0.04 对应占画面约 4%（全身帧约 15~40%）
        area_score = min(1.0, avg_max_area / 0.20)  # 0.20 = 20% 面积为满分参考
        score_components.append(("dominant_area", area_score, 0.20))
        if avg_max_area < self._cfg.min_dominant_area_ratio:
            reasons.append(
                f"WARN:low_dominant_area({avg_max_area:.3f} < {self._cfg.min_dominant_area_ratio:.3f})"
            )

        # 非多人帧惩罚（多人帧越少越好）
        non_crowd_score = 1.0 - crowded_ratio
        score_components.append(("non_crowded", non_crowd_score, 0.15))

        # 运动分（可选，权重较低）
        if motion_score is not None:
            score_components.append(("motion", motion_score, 0.10))
            # 补偿其他项满分权重到 0.90，此处直接使用 0.10 额外加成
            if motion_score < 0.2:
                reasons.append(f"WARN:low_motion_score({motion_score:.3f})—may_be_static_cover")
        else:
            # 无运动分时重新归一化权重（不改变相对比例，直接 normalize）
            pass

        # 归一化加权
        total_w = sum(w for _, _, w in score_components)
        raw_score = sum(s * w for _, s, w in score_components) / total_w
        score = round(float(raw_score), 4)

        # ---- 标签分配 ----
        if score >= self._cfg.solo_threshold:
            label = "likely_solo"
            reasons.append(f"PASS:solo_score={score:.3f} >= threshold={self._cfg.solo_threshold}")
        elif score >= self._cfg.uncertain_threshold:
            label = "uncertain"
            # 说明：节目类多机位视频容易落在此区间，应进 review
            reasons.append(
                f"UNCERTAIN:solo_score={score:.3f} in "
                f"[{self._cfg.uncertain_threshold:.2f}, {self._cfg.solo_threshold:.2f}). "
                f"May be multi-camera program—check audio_score before drop."
            )
        else:
            label = "likely_multi"
            reasons.append(
                f"REJECT:solo_score={score:.3f} < uncertain_threshold={self._cfg.uncertain_threshold}"
            )

        result.score = score
        result.label = label
        result.reasons = reasons

        logger.debug(
            "SoloScorer: score=%.3f label=%s single=%.2f crowded=%.2f valid=%.2f area=%.3f",
            score,
            label,
            single_ratio,
            crowded_ratio,
            valid_ratio,
            avg_max_area,
        )
        return result

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_motion_score(detections: list[FrameDetection]) -> Optional[float]:
        """通过主体人物 bbox 中心位置和面积变化估算运动程度。

        策略：相邻帧间最大 bbox 中心位移 + 面积变化的归一化标准差。
        只在有足够帧且有效人体帧时计算；否则返回 None。

        Returns:
            0~1 的运动估计分数；None 表示无法计算
        """
        # 只取单人帧的主体 bbox
        single_dets = [d for d in detections if d.person_count == 1 and d.bboxes]
        if len(single_dets) < 5:
            return None  # 样本不足

        centers_x, centers_y, areas = [], [], []
        for d in single_dets:
            b = d.bboxes[0]
            cx = (b.x1 + b.x2) / 2.0 / d.frame_w if d.frame_w > 0 else 0.5
            cy = (b.y1 + b.y2) / 2.0 / d.frame_h if d.frame_h > 0 else 0.5
            centers_x.append(cx)
            centers_y.append(cy)
            areas.append(b.area_ratio)

        # 相邻帧间位移
        diffs = []
        for i in range(1, len(centers_x)):
            dx = abs(centers_x[i] - centers_x[i - 1])
            dy = abs(centers_y[i] - centers_y[i - 1])
            diffs.append(math.hypot(dx, dy))

        mean_disp = float(np.mean(diffs)) if diffs else 0.0

        # 面积变化
        area_std = float(np.std(areas)) if areas else 0.0

        # 归一化：日常舞蹈 mean_disp ~ 0.02~0.08（帧间 1 秒时）
        motion_from_disp = min(1.0, mean_disp / 0.06)
        motion_from_area = min(1.0, area_std / 0.05)

        return round(0.7 * motion_from_disp + 0.3 * motion_from_area, 4)
