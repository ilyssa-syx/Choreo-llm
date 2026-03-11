"""
audio_quality_scorer.py
=======================
音频质量评分模块。

评估目标：筛选"音乐清晰、语音干扰少"的视频，作为编舞 LLM 训练的代理质量指标。

注意：
  - "音乐清晰"是训练友好的代理目标，不等价于版权可用性
  - 仍需后续人工与合规确认

评分模式（硬性筛选）：
  逐项检查以下指标，任一不达标立即拒绝：
  1. audio_present_ratio >= min_audio_present_ratio
  2. loudness_mean >= min_loudness_mean
  3. clipping_ratio <= max_clipping_ratio
  4. snr_proxy >= min_snr_proxy (可选)
  5. speech_ratio <= speech_ratio_max
  6. music_confidence >= min_music_conf (可选)
  
  所有检查通过 -> audio_score=1.0, label="clear_music"
  
  不再使用综合评分，而是"全通过"或"拒绝"的二元模式。

启发式指标（不需额外模型）：
  - 响度分析（音量、稳定性）
  - 静音检测（audio_present_ratio）
  - 削波检测（clipping_ratio）
  - 简易信噪比（snr_proxy）
  - 简易语音/音乐分离启发式（基于频谱特征）
  - 可选节拍强度（librosa.beat）

执行顺序（pipeline 中由 pipeline.py 调用）：
  1. 从本地视频文件提取音轨（ffmpeg）
  2. 依次检查各项指标
  3. 任一不通过立即返回拒绝结果
  4. 全部通过后输出 audio_score=1.0 + audio_label="clear_music"
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
        """加载音频并逐项检查质量指标（硬性筛选模式）。"""
        y, sr = self._load_audio(audio_path)
        if y is None or len(y) == 0:
            logger.warning("Empty audio signal for bvid=%s", bvid)
            return _default_result("empty_audio_signal")

        reasons: list[str] = []
        stats = AudioStats()
        cfg = self._cfg

        # ---- 1. 非静音占比 ----
        audio_present_ratio = self._calc_audio_present_ratio(y, sr)
        stats.audio_present_ratio = round(audio_present_ratio, 4)

        if audio_present_ratio < cfg.min_audio_present_ratio:
            reasons.append(
                f"REJECT:insufficient_audio({audio_present_ratio:.2f} "
                f"< {cfg.min_audio_present_ratio:.2f})"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="insufficient_audio",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 2. 平均响度 ----
        loudness_mean, loudness_stability = self._calc_loudness(y, sr)
        stats.loudness_mean = round(loudness_mean, 2)
        stats.loudness_stability = round(loudness_stability, 4)

        if loudness_mean < cfg.min_loudness_mean:
            reasons.append(
                f"REJECT:low_loudness({loudness_mean:.1f} dB < {cfg.min_loudness_mean:.1f} dB)"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="low_loudness",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 3. 削波比例 ----
        clipping_ratio = self._calc_clipping_ratio(y)
        stats.clipping_ratio = round(clipping_ratio, 4)

        if clipping_ratio > cfg.max_clipping_ratio:
            reasons.append(
                f"REJECT:high_clipping({clipping_ratio:.3f} > {cfg.max_clipping_ratio:.3f})"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="high_clipping",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 4. 信噪比 ----
        snr_proxy = self._calc_snr_proxy(y, sr)
        stats.snr_proxy = round(snr_proxy, 2)

        # 检查配置中是否有 min_snr_proxy（可选）
        min_snr = getattr(cfg, 'min_snr_proxy', None)
        if min_snr is not None and snr_proxy < min_snr:
            reasons.append(
                f"REJECT:low_snr({snr_proxy:.1f} dB < {min_snr:.1f} dB)"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="low_snr",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 5. 语音占比 ----
        speech_ratio = self._heuristic_speech_ratio(y, sr)
        stats.speech_ratio = round(speech_ratio, 4)

        if speech_ratio > cfg.speech_ratio_max:
            reasons.append(
                f"REJECT:speech_heavy({speech_ratio:.2f} > {cfg.speech_ratio_max:.2f})"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="speech_heavy",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 6. 音乐置信度 ----
        music_conf = self._heuristic_music_confidence(y, sr)
        stats.music_confidence = round(music_conf, 4)

        # 检查配置中是否有 min_music_conf（可选）
        min_music = getattr(cfg, 'min_music_conf', None)
        if min_music is not None and music_conf < min_music:
            reasons.append(
                f"REJECT:weak_music_evidence({music_conf:.2f} < {min_music:.2f})"
            )
            return AudioScoreResult(
                audio_score=0.0,
                audio_label="weak_music_evidence",
                audio_reasons=reasons,
                audio_stats=stats,
            )

        # ---- 7. 节拍强度（可选，仅统计）----
        if _LIBROSA_AVAILABLE:
            beat_strength = self._calc_beat_strength(y, sr)
            stats.beat_strength = round(beat_strength, 4) if beat_strength is not None else None
        else:
            stats.beat_strength = None

        # ---- 所有检查通过 ----
        reasons.append(
            f"PASS:all_checks_passed(present={audio_present_ratio:.2f}, "
            f"loudness={loudness_mean:.1f}dB, clipping={clipping_ratio:.3f}, "
            f"snr={snr_proxy:.1f}dB, speech={speech_ratio:.2f}, music={music_conf:.2f})"
        )

        logger.info(
            "AudioScorer bvid=%s: PASS label=clear_music speech=%.2f music=%.2f snr=%.1f",
            bvid,
            speech_ratio,
            music_conf,
            snr_proxy,
        )

        return AudioScoreResult(
            audio_score=1.0,
            audio_label="clear_music",
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
