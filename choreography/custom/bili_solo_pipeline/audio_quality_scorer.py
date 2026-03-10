"""
audio_quality_scorer.py
=======================
音频质量评分模块。

评估目标：筛选"音乐清晰、语音干扰少"的视频，作为编舞 LLM 训练的代理质量指标。

注意：
  - "音乐清晰"是训练友好的代理目标，不等价于版权可用性
  - 仍需后续人工与合规确认

两种模式（自动切换）：
  A. 启发式模式（use_heuristic=True, 默认）
     仅依赖 ffmpeg + librosa/pydub，不需额外模型
     - 响度分析（音量、稳定性）
     - 静音检测（audio_present_ratio）
     - 削波检测（clipping_ratio）
     - 简易语音/音乐分离启发式（基于频谱特征）
     - 可选节拍强度（librosa.beat）

  B. 模型模式（use_heuristic=False）
     使用音频分类模型（如 panns_inference / speechbrain）区分 speech vs music
     若模型不可用自动降级到 A 模式

执行顺序（pipeline 中由 pipeline.py 调用）：
  1. 从本地视频文件提取音轨（ffmpeg）
  2. 计算各项指标
  3. 综合评分输出 audio_score + audio_label + audio_reasons
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .models import AudioStats
from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import librosa  # type: ignore
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.info("librosa not installed; beat detection will be disabled.")

try:
    import soundfile as sf  # type: ignore
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    _SOUNDFILE_AVAILABLE = False

# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------

@dataclass
class AudioScoreResult:
    audio_score: float = 0.0
    audio_label: str = "uncertain_audio"
    audio_reasons: list[str] = field(default_factory=list)
    audio_stats: Optional[AudioStats] = None


# ---------------------------------------------------------------------------
# 主评分器
# ---------------------------------------------------------------------------

class AudioQualityScorer:
    """音频质量评分器。

    Args:
        config: 全局配置
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG) -> None:
        self._cfg = config
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        if not self._ffmpeg_available:
            logger.warning(
                "ffmpeg not found. Audio extraction will be skipped "
                "and all videos will receive audio_label='uncertain_audio'."
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def score_from_video(
        self,
        video_path: str | Path,
        bvid: str = "",
    ) -> AudioScoreResult:
        """从本地视频文件提取音轨并评分。

        Args:
            video_path: 本地视频文件路径
            bvid: 视频 ID（仅用于日志）

        Returns:
            AudioScoreResult
        """
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning("Video not found for audio analysis: %s", video_path)
            return _default_result("video_file_missing")

        if not self._ffmpeg_available:
            return _default_result("ffmpeg_unavailable")

        # 提取音轨到临时 wav 文件
        with tempfile.TemporaryDirectory(prefix="bili_audio_") as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            success = self._extract_audio(str(video_path), audio_path, bvid)
            if not success:
                return _default_result("audio_extraction_failed")
            return self._analyze_audio(audio_path, bvid)

    def score_from_audio(self, audio_path: str | Path, bvid: str = "") -> AudioScoreResult:
        """从已提取的音频文件评分（支持 wav/mp3/flac 等）。"""
        if not Path(audio_path).exists():
            return _default_result("audio_file_missing")
        return self._analyze_audio(str(audio_path), bvid)

    # ------------------------------------------------------------------
    # 音轨提取
    # ------------------------------------------------------------------

    def _extract_audio(
        self,
        video_path: str,
        out_wav: str,
        bvid: str = "",
    ) -> bool:
        """使用 ffmpeg 从视频中提取单声道 22050 Hz wav 片段。

        取视频中段（避免片头/片尾安静段影响评估）。
        """
        duration = self._cfg.audio_segment_duration_sec
        # 先获取视频时长，取中段
        total_sec = self._get_video_duration(video_path)
        if total_sec <= 0:
            start = 0.0
        elif total_sec <= duration:
            start = 0.0
        else:
            # 取 1/4 处开始到 1/4+duration 处
            start = max(0.0, total_sec * 0.25)
            if start + duration > total_sec:
                start = max(0.0, total_sec - duration)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", video_path,
            "-vn",                       # 不要视频流
            "-ac", "1",                  # 单声道
            "-ar", "22050",              # 22050 Hz（librosa 默认 sr）
            "-f", "wav",
            out_wav,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "ffmpeg audio extraction failed (bvid=%s): %s",
                    bvid,
                    result.stderr[-300:],
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timeout for bvid=%s", bvid)
            return False
        except Exception as exc:
            logger.error("ffmpeg error for bvid=%s: %s", bvid, exc)
            return False

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """通过 ffprobe 获取视频时长。"""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 音频分析（启发式）
    # ------------------------------------------------------------------

    def _analyze_audio(self, audio_path: str, bvid: str = "") -> AudioScoreResult:
        """加载音频并计算各项指标（启发式模式）。"""
        y, sr = self._load_audio(audio_path)
        if y is None or len(y) == 0:
            logger.warning("Empty audio signal for bvid=%s", bvid)
            return _default_result("empty_audio_signal")

        reasons: list[str] = []
        stats = AudioStats()

        # ---- 1. 非静音占比 ----
        audio_present_ratio = self._calc_audio_present_ratio(y, sr)
        stats.audio_present_ratio = round(audio_present_ratio, 4)

        if audio_present_ratio < self._cfg.min_audio_present_ratio:
            reasons.append(
                f"REJECT:low_audio_present_ratio({audio_present_ratio:.2f} "
                f"< {self._cfg.min_audio_present_ratio:.2f})"
            )
            result = AudioScoreResult(
                audio_score=0.1,
                audio_label="low_volume",
                audio_reasons=reasons,
                audio_stats=stats,
            )
            return result

        # ---- 2. 平均响度（LUFS 近似 via RMS）----
        loudness_mean, loudness_stability = self._calc_loudness(y, sr)
        stats.loudness_mean = round(loudness_mean, 2)
        stats.loudness_stability = round(loudness_stability, 4)

        if loudness_mean < self._cfg.min_loudness_mean:
            reasons.append(
                f"WARN:low_loudness({loudness_mean:.1f} dB < {self._cfg.min_loudness_mean:.1f} dB)"
            )

        # ---- 3. 削波比例 ----
        clipping_ratio = self._calc_clipping_ratio(y)
        stats.clipping_ratio = round(clipping_ratio, 4)
        if clipping_ratio > self._cfg.max_clipping_ratio:
            reasons.append(f"WARN:high_clipping({clipping_ratio:.3f})")

        # ---- 4. 简易信噪比代理 ----
        snr_proxy = self._calc_snr_proxy(y, sr)
        stats.snr_proxy = round(snr_proxy, 2)

        # ---- 5. 语音占比（启发式）----
        speech_ratio = self._heuristic_speech_ratio(y, sr)
        stats.speech_ratio = round(speech_ratio, 4)
        if speech_ratio > self._cfg.speech_ratio_max:
            reasons.append(
                f"WARN:high_speech_ratio({speech_ratio:.2f} > {self._cfg.speech_ratio_max:.2f})—"
                f"may_contain_commentary/interview"
            )

        # ---- 6. 音乐置信度（启发式）----
        music_conf = self._heuristic_music_confidence(y, sr)
        stats.music_confidence = round(music_conf, 4)

        # ---- 7. 节拍强度（可选，需 librosa）----
        if _LIBROSA_AVAILABLE:
            beat_strength = self._calc_beat_strength(y, sr)
            stats.beat_strength = round(beat_strength, 4) if beat_strength is not None else None
        else:
            stats.beat_strength = None

        # ---- 综合评分 ----
        audio_score, audio_label = self._compute_audio_score(
            audio_present_ratio=audio_present_ratio,
            speech_ratio=speech_ratio,
            music_conf=music_conf,
            snr_proxy=snr_proxy,
            clipping_ratio=clipping_ratio,
            loudness_mean=loudness_mean,
            loudness_stability=loudness_stability,
        )
        reasons.append(f"audio_score={audio_score:.3f} label={audio_label}")

        logger.debug(
            "AudioScorer bvid=%s: score=%.3f label=%s speech=%.2f music=%.2f snr=%.1f",
            bvid,
            audio_score,
            audio_label,
            speech_ratio,
            music_conf,
            snr_proxy,
        )

        return AudioScoreResult(
            audio_score=round(audio_score, 4),
            audio_label=audio_label,
            audio_reasons=reasons,
            audio_stats=stats,
        )

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: str) -> tuple[Optional[np.ndarray], int]:
        """加载 wav 音频，返回 (samples, sample_rate)。"""
        if _LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                return y.astype(np.float32), int(sr)
            except Exception as exc:
                logger.warning("librosa load failed: %s", exc)

        if _SOUNDFILE_AVAILABLE:
            try:
                y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                return y, int(sr)
            except Exception as exc:
                logger.warning("soundfile load failed: %s", exc)

        # 最后手段：scipy
        try:
            from scipy.io import wavfile  # type: ignore
            sr, y = wavfile.read(audio_path)
            y = y.astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y / (np.abs(y).max() + 1e-8)
            return y, int(sr)
        except Exception as exc:
            logger.error("Audio load failed: %s", exc)
            return None, 16000

    @staticmethod
    def _calc_audio_present_ratio(y: np.ndarray, sr: int, frame_len: float = 0.02) -> float:
        """非静音帧占比（RMS 阈值法）。

        使用固定绝对阈值 (~-40 dBFS) 判断静音，避免自适应阈值在稳定信号上误判。
        """
        frame_size = max(1, int(frame_len * sr))
        hop_size = frame_size
        rms_frames = []
        for i in range(0, len(y) - frame_size, hop_size):
            rms = float(np.sqrt(np.mean(y[i: i + frame_size] ** 2) + 1e-12))
            rms_frames.append(rms)
        if not rms_frames:
            return 0.0
        # 固定阈值 0.01 ≈ -40 dBFS (RMS)
        threshold = 0.01
        nonsilent = sum(1 for r in rms_frames if r > threshold)
        return nonsilent / len(rms_frames)

    @staticmethod
    def _calc_loudness(
        y: np.ndarray, sr: int, frame_sec: float = 0.5
    ) -> tuple[float, float]:
        """计算平均响度（dB RMS 近似 LUFS）与稳定性。

        Returns:
            (loudness_mean_dbrms, loudness_stability)
            稳定性 = 1 - normalized_std（越接近 1 越稳定）
        """
        frame_size = max(1, int(frame_sec * sr))
        energies_db = []
        for i in range(0, len(y) - frame_size, frame_size):
            rms = np.sqrt(np.mean(y[i: i + frame_size] ** 2) + 1e-12)
            db = 20 * np.log10(rms)
            energies_db.append(db)
        if not energies_db:
            return -60.0, 0.0
        mean_db = float(np.mean(energies_db))
        std_db = float(np.std(energies_db))
        # 响度范围通常 -60 ~ 0 dB，std 归一化到 20 dB 范围
        stability = max(0.0, 1.0 - std_db / 20.0)
        return mean_db, stability

    @staticmethod
    def _calc_clipping_ratio(y: np.ndarray, threshold: float = 0.98) -> float:
        """削波比例：绝对值接近 1 的样本占比。"""
        return float(np.mean(np.abs(y) >= threshold))

    @staticmethod
    def _calc_snr_proxy(y: np.ndarray, sr: int) -> float:
        """简易信噪比代理（dB）。

        方法：将信号分为高能量帧（信号+噪声）和低能量帧（噪底），
        用两者 RMS 差估算 SNR。
        """
        frame_size = max(1, int(0.05 * sr))  # 50ms 帧
        rms_frames = []
        for i in range(0, len(y) - frame_size, frame_size):
            rms = float(np.sqrt(np.mean(y[i: i + frame_size] ** 2) + 1e-12))
            rms_frames.append(rms)
        if len(rms_frames) < 4:
            return 0.0
        rms_sorted = sorted(rms_frames)
        noise_floor = float(np.mean(rms_sorted[:max(1, len(rms_sorted) // 10)]))
        signal_peak = float(np.mean(rms_sorted[-max(1, len(rms_sorted) // 10):]))
        if noise_floor < 1e-10:
            return 60.0
        snr = 20 * np.log10(signal_peak / noise_floor)
        return float(snr)

    @staticmethod
    def _heuristic_speech_ratio(y: np.ndarray, sr: int) -> float:
        """启发式语音占比估算。

        原理：
          - 语音主要能量集中在 300~3400 Hz（基频 + 共振峰）
          - 音乐能量分布更宽（低频打击乐 + 高频泛音）
          - 使用频谱能量比率作为粗粒度代理

        局限性：此方法为粗粒度启发式，准确率有限；
        若需精确区分，请使用 panns_inference 或 speechbrain。
        """
        if not _LIBROSA_AVAILABLE:
            return 0.0  # 无法计算，返回中性值

        try:
            # STFT
            n_fft = 1024
            hop_length = 512
            stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # 语音频带：300~3400 Hz
            speech_mask = (freqs >= 300) & (freqs <= 3400)
            # 全频段
            total_energy = np.sum(stft ** 2, axis=0) + 1e-10
            speech_energy = np.sum(stft[speech_mask, :] ** 2, axis=0)
            ratio_per_frame = speech_energy / total_energy  # (T,)

            # 对于纯语音帧，这个比率应该很高（>0.7）
            # 对于音乐帧，低频和高频都有贡献，比率较低（<0.5）
            # 阈值可调
            speech_frame_ratio = float(np.mean(ratio_per_frame > 0.65))
            return speech_frame_ratio
        except Exception as exc:
            logger.debug("Heuristic speech ratio failed: %s", exc)
            return 0.0

    @staticmethod
    def _heuristic_music_confidence(y: np.ndarray, sr: int) -> float:
        """启发式音乐置信度。

        基于频谱通量和零交叉率的简单组合：
          - 音乐通常具有更规律的零交叉率和频谱通量变化
          - 语音具有较高的短时频谱变化

        Returns:
            0~1 的音乐置信度
        """
        if not _LIBROSA_AVAILABLE:
            return 0.5  # 无法判断，返回中性值

        try:
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))

            # 频谱质心（音乐通常更一致）
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
            centroid_std_norm = float(np.std(centroid) / (np.mean(centroid) + 1e-8))

            # 频谱通量（相邻帧能量差）
            stft = np.abs(librosa.stft(y, hop_length=512))
            flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
            flux_mean = float(np.mean(flux))
            flux_std = float(np.std(flux))
            flux_cv = flux_std / (flux_mean + 1e-8)  # 变异系数

            # 组合：音乐特征 -> 低 zcr_std、适中 zcr_mean、低 centroid_std_norm
            music_conf = 0.5
            # zcr：语音约 0.05~0.15，音乐约 0.03~0.10（重叠区域多）
            if zcr_std < 0.03:
                music_conf += 0.15  # 稳定 zcr -> 偏音乐
            if centroid_std_norm < 0.3:
                music_conf += 0.15  # 稳定频谱质心 -> 偏音乐
            if flux_cv < 0.8:
                music_conf += 0.10  # 相对规律的通量变化 -> 偏音乐
            if zcr_mean > 0.12:
                music_conf -= 0.15  # 高 zcr -> 偏语音

            return float(min(1.0, max(0.0, music_conf)))
        except Exception as exc:
            logger.debug("Heuristic music confidence failed: %s", exc)
            return 0.5

    @staticmethod
    def _calc_beat_strength(y: np.ndarray, sr: int) -> Optional[float]:
        """使用 librosa 估计节拍强度（normalize 到 0~1）。"""
        if not _LIBROSA_AVAILABLE:
            return None
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            # 节拍强度用 onset_strength 均值代理
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            strength = float(np.mean(onset_env))
            # 归一化：strength 通常在 0~5 之间
            return min(1.0, strength / 3.0)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 综合评分
    # ------------------------------------------------------------------

    def _compute_audio_score(
        self,
        audio_present_ratio: float,
        speech_ratio: float,
        music_conf: float,
        snr_proxy: float,
        clipping_ratio: float,
        loudness_mean: float,
        loudness_stability: float,
    ) -> tuple[float, str]:
        """综合所有指标输出 audio_score 和 audio_label。

        Returns:
            (score, label)
        """
        cfg = self._cfg
        score = 0.5  # 基础分

        # 音乐置信度加分
        score += 0.25 * music_conf

        # 语音占比惩罚
        if speech_ratio > cfg.speech_ratio_max:
            penalty = min(0.4, (speech_ratio - cfg.speech_ratio_max) * 1.2)
            score -= penalty

        # SNR 加分（越高越好，30 dB 为参考满分）
        snr_score = min(1.0, max(0.0, snr_proxy / 30.0))
        score += 0.10 * snr_score

        # 削波惩罚
        if clipping_ratio > cfg.max_clipping_ratio:
            score -= min(0.2, clipping_ratio * 5)

        # 响度合理范围 -35 ~ -15 dBRMS
        if -35 <= loudness_mean <= -5:
            score += 0.05
        elif loudness_mean < cfg.min_loudness_mean:
            score -= 0.15

        # 稳定性加分
        score += 0.05 * loudness_stability

        # 非静音占比
        score += 0.05 * audio_present_ratio

        score = round(float(min(1.0, max(0.0, score))), 4)

        # ---- 标签分配 ----
        # 语音超过阈值且占主导（>= speech_ratio_max）-> speech_heavy
        if speech_ratio >= cfg.speech_ratio_max and speech_ratio >= 0.35:
            label = "speech_heavy"
        elif audio_present_ratio < 0.3 or loudness_mean < cfg.min_loudness_mean:
            label = "low_volume"
        elif clipping_ratio > cfg.max_clipping_ratio * 3:
            label = "noisy_audio"
        elif score >= cfg.audio_threshold:
            label = "clear_music"
        else:
            label = "uncertain_audio"

        return score, label


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _default_result(reason: str) -> AudioScoreResult:
    return AudioScoreResult(
        audio_score=0.0,
        audio_label="uncertain_audio",
        audio_reasons=[f"SKIP:{reason}"],
        audio_stats=AudioStats(),
    )
