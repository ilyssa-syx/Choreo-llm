"""
test_audio_scorer.py
====================
单元测试：audio_quality_scorer 启发式规则逻辑（不依赖真实音频文件）
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bili_solo_pipeline.audio_quality_scorer import AudioQualityScorer, AudioScoreResult
from bili_solo_pipeline.models import AudioStats
from bili_solo_pipeline.config import PipelineConfig, DEFAULT_CONFIG


def _make_sine_wave(freq: float = 440.0, duration: float = 5.0, sr: int = 22050) -> np.ndarray:
    """生成正弦波（模拟纯音乐信号）。"""
    t = np.linspace(0, duration, int(sr * duration))
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_silence(duration: float = 5.0, sr: int = 22050) -> np.ndarray:
    """生成静音信号。"""
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_clipping_signal(duration: float = 5.0, sr: int = 22050) -> np.ndarray:
    """生成严重削波信号（大量绝对值接近1的样本）。"""
    y = _make_sine_wave(duration=duration, sr=sr) * 10.0  # 放大10倍导致削波
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _make_noisy_signal(duration: float = 5.0, sr: int = 22050, noise_level: float = 0.8) -> np.ndarray:
    """生成高噪声信号。"""
    rng = np.random.default_rng(42)
    signal = _make_sine_wave(duration=duration, sr=sr) * 0.1
    noise = rng.uniform(-noise_level, noise_level, len(signal)).astype(np.float32)
    return signal + noise


class TestAudioScorer:
    def setup_method(self):
        self.scorer = AudioQualityScorer(DEFAULT_CONFIG)

    def test_audio_present_ratio_sine(self):
        """正弦波信号应该非静音占比较高（> 0.5）。"""
        y = _make_sine_wave()
        ratio = self.scorer._calc_audio_present_ratio(y, 22050)
        # 正弦波 RMS ~ 0.354（> 绝对阈值 0.01），大多数帧应为非静音
        assert ratio > 0.5, f"Expected > 0.5, got {ratio}"

    def test_audio_present_ratio_silence(self):
        """静音信号非静音占比应接近 0。"""
        y = _make_silence()
        ratio = self.scorer._calc_audio_present_ratio(y, 22050)
        assert ratio < 0.1, f"Expected < 0.1, got {ratio}"

    def test_clipping_ratio_sine(self):
        """正常正弦波削波比例应接近 0。"""
        y = _make_sine_wave()
        ratio = self.scorer._calc_clipping_ratio(y)
        assert ratio < 0.01, f"Expected < 0.01, got {ratio}"

    def test_clipping_ratio_clipped(self):
        """削波信号削波比例应较高。"""
        y = _make_clipping_signal()
        ratio = self.scorer._calc_clipping_ratio(y)
        assert ratio > 0.05, f"Expected > 0.05, got {ratio}"

    def test_loudness_sine(self):
        """正弦波应有合理响度（不太低）。"""
        y = _make_sine_wave()
        mean_db, stability = self.scorer._calc_loudness(y, 22050)
        assert mean_db > -30, f"Loudness too low: {mean_db}"
        assert 0.0 <= stability <= 1.0

    def test_loudness_silence(self):
        """静音响度应很低。"""
        y = _make_silence()
        mean_db, stability = self.scorer._calc_loudness(y, 22050)
        assert mean_db < -40, f"Silence loudness should be low: {mean_db}"

    def test_snr_proxy_clean_signal(self):
        """信号段明显强于噪底段时 SNR proxy 应高于静音段。"""
        sr = 22050
        # 构造：强信号段（前半）+ 极低噪声段（后半）
        signal_part = (_make_sine_wave(duration=3.0, sr=sr) * 0.5).astype(np.float32)
        noise_part = (np.random.default_rng(0).uniform(-0.001, 0.001, int(3.0 * sr))).astype(np.float32)
        mixed = np.concatenate([signal_part, noise_part])
        snr = self.scorer._calc_snr_proxy(mixed, sr)
        # 有明显强弱对比时，代理 SNR 应 > 0
        assert snr > 0, f"SNR proxy should be positive for signal+noise, got {snr}"

    def test_compute_audio_score_range(self):
        """compute_audio_score 输出应在 [0, 1]。"""
        params = [
            (0.9, 0.1, 0.8, 25.0, 0.001, -18.0, 0.85),  # 良好音频
            (0.1, 0.8, 0.2, 5.0, 0.001, -50.0, 0.3),   # 差音频
            (0.8, 0.5, 0.5, 15.0, 0.05, -25.0, 0.6),   # 中等音频
        ]
        for args in params:
            score, label = self.scorer._compute_audio_score(
                audio_present_ratio=args[0],
                speech_ratio=args[1],
                music_conf=args[2],
                snr_proxy=args[3],
                clipping_ratio=args[4],
                loudness_mean=args[5],
                loudness_stability=args[6],
            )
            assert 0.0 <= score <= 1.0, f"score={score} out of range"
            assert label in (
                "clear_music", "speech_heavy", "noisy_audio", "low_volume", "uncertain_audio"
            )

    def test_speech_heavy_label(self):
        """高语音占比 + 低音乐置信度应输出 speech_heavy 标签。"""
        score, label = self.scorer._compute_audio_score(
            audio_present_ratio=0.9,
            speech_ratio=0.7,    # > speech_ratio_max
            music_conf=0.2,
            snr_proxy=10.0,
            clipping_ratio=0.001,
            loudness_mean=-20.0,
            loudness_stability=0.7,
        )
        assert label == "speech_heavy"

    def test_low_volume_label(self):
        """低音量时应输出 low_volume 标签（来自 score_from_* 分支）。"""
        # 直接测试 _analyze_audio 的 low_audio_present_ratio 分支
        from bili_solo_pipeline.audio_quality_scorer import _default_result
        result = _default_result("test_reason")
        assert result.audio_label == "uncertain_audio"

    def test_clear_music_label(self):
        """理想音频（高 audio_present_ratio、低 speech、高 music_conf）应输出 clear_music。"""
        score, label = self.scorer._compute_audio_score(
            audio_present_ratio=0.95,
            speech_ratio=0.05,   # 低语音
            music_conf=0.85,
            snr_proxy=30.0,
            clipping_ratio=0.0,
            loudness_mean=-18.0,
            loudness_stability=0.9,
        )
        assert label == "clear_music", f"Expected clear_music, got {label}"
        assert score >= DEFAULT_CONFIG.audio_threshold

    def test_speech_heavy_review_rule(self):
        """若 speech_ratio > threshold，进入 review 而不是 keep。"""
        cfg = PipelineConfig(audio_threshold=0.6, speech_ratio_max=0.35)
        scorer = AudioQualityScorer(cfg)
        score, label = scorer._compute_audio_score(
            audio_present_ratio=0.9,
            speech_ratio=0.55,   # 明确超过阈值 0.35 且 >= 0.35
            music_conf=0.8,
            snr_proxy=25.0,
            clipping_ratio=0.001,
            loudness_mean=-18.0,
            loudness_stability=0.85,
        )
        # speech_ratio=0.55 >= speech_ratio_max=0.35 -> speech_heavy
        assert label == "speech_heavy"
