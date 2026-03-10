"""
test_solo_scorer.py
===================
单元测试：solo_scorer 规则逻辑
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bili_solo_pipeline.solo_scorer import SoloScorer
from bili_solo_pipeline.person_detector import FrameDetection, BBox
from bili_solo_pipeline.config import PipelineConfig


def _make_det(
    person_count: int,
    max_area_ratio: float = 0.15,
    frame_w: int = 1920,
    frame_h: int = 1080,
    timestamp: float = 0.0,
) -> FrameDetection:
    """创建模拟的 FrameDetection 对象。"""
    det = FrameDetection(
        timestamp=timestamp,
        frame_w=frame_w,
        frame_h=frame_h,
        person_count=person_count,
    )
    if person_count > 0:
        # 生成对应数量的 bbox
        bboxes = []
        for i in range(person_count):
            bboxes.append(BBox(
                x1=0.0, y1=0.0,
                x2=frame_w * (max_area_ratio ** 0.5),
                y2=frame_h * (max_area_ratio ** 0.5),
                conf=0.9,
                area_ratio=max_area_ratio if i == 0 else max_area_ratio * 0.3,
            ))
        det.bboxes = bboxes
        det.max_bbox_area_ratio = bboxes[0].area_ratio
    return det


def _make_detections_all_single(n: int = 30) -> list[FrameDetection]:
    """创建全部为单人的帧序列。"""
    return [_make_det(1, max_area_ratio=0.20, timestamp=float(i)) for i in range(n)]


def _make_detections_all_multi(n: int = 30) -> list[FrameDetection]:
    """创建全部为多人的帧序列。"""
    return [_make_det(3, max_area_ratio=0.05, timestamp=float(i)) for i in range(n)]


def _make_detections_no_person(n: int = 30) -> list[FrameDetection]:
    """创建全部无人的帧序列。"""
    return [_make_det(0, timestamp=float(i)) for i in range(n)]


def _make_detections_mixed(n_single: int, n_multi: int, n_empty: int = 0) -> list[FrameDetection]:
    """创建混合帧序列。"""
    dets = []
    for i in range(n_single):
        dets.append(_make_det(1, max_area_ratio=0.20, timestamp=float(i)))
    for i in range(n_multi):
        dets.append(_make_det(2, max_area_ratio=0.06, timestamp=float(n_single + i)))
    for i in range(n_empty):
        dets.append(_make_det(0, timestamp=float(n_single + n_multi + i)))
    return dets


class TestSoloScorerBasic:
    def setup_method(self):
        self.scorer = SoloScorer()

    def test_empty_detections(self):
        result = self.scorer.score([])
        assert result.label == "no_person"
        assert result.score < 0.1

    def test_all_single_person(self):
        dets = _make_detections_all_single(30)
        result = self.scorer.score(dets)
        assert result.label == "likely_solo"
        assert result.score >= 0.75  # 应达到 solo_threshold

    def test_all_multi_person(self):
        dets = _make_detections_all_multi(30)
        result = self.scorer.score(dets)
        assert result.label in ("likely_multi",)
        assert result.score < 0.5

    def test_no_person_frames(self):
        dets = _make_detections_no_person(30)
        result = self.scorer.score(dets)
        assert result.label == "no_person"
        assert result.score < 0.2

    def test_mostly_single_some_empty(self):
        """90% 单人 + 10% 空帧 -> likely_solo"""
        dets = _make_detections_mixed(27, 0, 3)
        result = self.scorer.score(dets)
        assert result.label == "likely_solo"

    def test_high_crowded_ratio(self):
        """多人帧超过阈值 -> likely_multi"""
        dets = _make_detections_mixed(10, 20, 0)  # 33% 单人，67% 多人
        result = self.scorer.score(dets)
        assert result.label == "likely_multi"
        assert any("crowded_ratio" in r for r in result.reasons)


class TestSoloScorerStats:
    def setup_method(self):
        self.scorer = SoloScorer()

    def test_person_stats_fields(self):
        """person_stats 字段应正确填充。"""
        dets = _make_detections_all_single(20)
        result = self.scorer.score(dets)
        assert result.person_stats is not None
        stats = result.person_stats
        assert stats.single_person_frame_ratio == pytest.approx(1.0, abs=0.01)
        assert stats.valid_person_frame_ratio == pytest.approx(1.0, abs=0.01)
        assert stats.crowded_frame_ratio == pytest.approx(0.0, abs=0.01)

    def test_person_stats_mixed(self):
        dets = _make_detections_mixed(15, 10, 5)  # 30帧：15单人/10多人/5空
        result = self.scorer.score(dets)
        stats = result.person_stats
        assert stats is not None
        assert abs(stats.single_person_frame_ratio - 15 / 30) < 0.01
        assert abs(stats.crowded_frame_ratio - 10 / 30) < 0.01
        assert abs(stats.valid_person_frame_ratio - 25 / 30) < 0.01

    def test_score_in_valid_range(self):
        """各场景下 score 都应在 [0, 1]。"""
        scenarios = [
            _make_detections_all_single(),
            _make_detections_all_multi(),
            _make_detections_no_person(),
            _make_detections_mixed(15, 10, 5),
            [],
        ]
        for dets in scenarios:
            result = self.scorer.score(dets)
            assert 0.0 <= result.score <= 1.0, (
                f"score={result.score} out of range, dets_len={len(dets)}"
            )

    def test_uncertain_zone(self):
        """约 50% 单人 + 40% 多人 + 10% 空 -> uncertain 区间"""
        dets = _make_detections_mixed(15, 12, 3)  # 50%/40%/10%
        result = self.scorer.score(dets)
        cfg = self.scorer._cfg
        # 应该在 uncertain 或 likely_multi（取决于 crowded_ratio 是否超阈值）
        assert result.label in ("uncertain", "likely_multi")


class TestSoloScorerConfigurable:
    def test_custom_threshold(self):
        """提高 solo_threshold 后，原来的 likely_solo 可能变 uncertain。"""
        strict_cfg = PipelineConfig(solo_threshold=0.95, uncertain_threshold=0.45)
        strict_scorer = SoloScorer(strict_cfg)

        # 90% 单人 + 10% 空
        dets = _make_detections_mixed(27, 0, 3)
        result = strict_scorer.score(dets)
        # 在严格阈值下，90% 单人可能不够 0.95
        # score 应在 uncertain 或 likely_solo 之间，看实际分值
        assert result.label in ("likely_solo", "uncertain")

    def test_low_min_valid_person_threshold(self):
        """降低 min_valid_person_ratio 后，稀疏有人帧不再是 no_person。"""
        lenient_cfg = PipelineConfig(min_valid_person_ratio=0.1)
        lenient_scorer = SoloScorer(lenient_cfg)
        # 20% 帧有人（= 6/30）
        dets = _make_detections_mixed(6, 0, 24)
        result = lenient_scorer.score(dets)
        # valid_ratio=0.2 >= 0.1，不应是 no_person
        assert result.label != "no_person"
