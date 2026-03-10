"""
frame_sampler.py
================
从本地视频文件或视频流中按固定间隔抽帧。

设计说明：
  - 主路径：从本地已下载的视频文件抽帧（cv2.VideoCapture）
  - 若视频文件不存在：
      优先尝试通过 yt-dlp 拉取低分辨率视频流（可选，需用户手动启用）
      若均不可用，返回空列表并记录日志（供后续人工处理）
  - 跳过片头（skip_head_sec）以减少 Logo/题字干扰
  - 支持最大抽帧数（max_frames）避免太慢
  - 返回 (timestamps, frames) 列表，frames 为 numpy ndarray（BGR）

重要：
  - 本模块不自动批量下载视频，yt-dlp 调用为可选且需用户显式启用
  - 请遵守 Bilibili 服务条款，不进行大规模版权内容的自动下载
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("opencv-python not installed. Frame sampling will be unavailable.")


class FrameSampler:
    """从视频文件中按间隔抽帧。

    Args:
        sample_every_sec: 抽帧间隔（秒）
        max_frames: 最大抽帧数
        skip_head_sec: 跳过开头秒数（减少片头/Logo 干扰）
        enable_yt_dlp: 是否允许通过 yt-dlp 下载低分辨率视频进行检测
                       （默认关闭；启用前请确认符合平台服务条款）
        yt_dlp_format: yt-dlp 视频格式选择器（建议低分辨率节省带宽）
    """

    def __init__(
        self,
        sample_every_sec: float = 1.0,
        max_frames: int = 60,
        skip_head_sec: float = 3.0,
        enable_yt_dlp: bool = False,
        yt_dlp_format: str = "worst[ext=mp4]/worst",
    ) -> None:
        if not _CV2_AVAILABLE:
            logger.error(
                "opencv-python is required for frame sampling. "
                "Install with: pip install opencv-python-headless"
            )
        self.sample_every_sec = sample_every_sec
        self.max_frames = max_frames
        self.skip_head_sec = skip_head_sec
        self.enable_yt_dlp = enable_yt_dlp
        self.yt_dlp_format = yt_dlp_format

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def sample_from_file(
        self,
        video_path: str | Path,
    ) -> tuple[list[float], list[np.ndarray]]:
        """从本地视频文件抽帧。

        Args:
            video_path: 本地视频路径

        Returns:
            (timestamps_sec, frames) 两个等长列表；
            frames 中每个元素是 BGR numpy array (H, W, 3)
        """
        if not _CV2_AVAILABLE:
            return [], []

        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning("Video file not found: %s", video_path)
            return [], []

        return self._extract_frames_cv2(str(video_path))

    def sample_from_url(
        self,
        url: str,
        cache_dir: Optional[str | Path] = None,
        bvid: str = "",
    ) -> tuple[list[float], list[np.ndarray]]:
        """尝试从 URL 抽帧（需要 yt-dlp 且用户显式启用）。

        若 enable_yt_dlp=False，仅记录日志并返回空结果。
        使用者请确认此操作符合相关平台条款。

        Args:
            url: 视频页面 URL
            cache_dir: 下载缓存目录（None 使用临时目录）
            bvid: 视频 ID（用于缓存文件命名）

        Returns:
            同 sample_from_file 格式
        """
        if not self.enable_yt_dlp:
            logger.info(
                "yt-dlp download disabled. Provide local video file for frame sampling. "
                "bvid=%s url=%s",
                bvid,
                url,
            )
            return [], []

        if not _CV2_AVAILABLE:
            return [], []

        # 检查 yt-dlp 是否可用
        import shutil
        if not shutil.which("yt-dlp"):
            logger.error(
                "yt-dlp not found in PATH. Install with: pip install yt-dlp. "
                "Video URL: %s",
                url,
            )
            return [], []

        # 确定缓存路径
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_template = str(cache_dir / f"{bvid or 'video'}.%(ext)s")
            # 检查是否已有缓存
            for ext in ["mp4", "webm", "flv", "mkv"]:
                cached = cache_dir / f"{bvid or 'video'}.{ext}"
                if cached.exists():
                    logger.info("Using cached video: %s", cached)
                    return self.sample_from_file(cached)
        else:
            tmp_dir = tempfile.mkdtemp(prefix="bili_frame_")
            out_template = os.path.join(tmp_dir, f"{bvid or 'video'}.%(ext)s")

        logger.info("Downloading video for frame sampling: %s", url)
        cmd = [
            "yt-dlp",
            "--format", self.yt_dlp_format,
            "--no-playlist",
            "--no-part",
            "-o", out_template,
            url,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error("yt-dlp failed for %s:\n%s", url, result.stderr[-500:])
                return [], []
        except subprocess.TimeoutExpired:
            logger.error("yt-dlp timeout for %s", url)
            return [], []
        except FileNotFoundError:
            logger.error("yt-dlp not found.")
            return [], []

        # 找到下载的文件
        for ext in ["mp4", "webm", "flv", "mkv"]:
            video_file = Path(out_template.replace("%(ext)s", ext))
            if video_file.exists():
                return self.sample_from_file(video_file)

        logger.warning("Downloaded file not found after yt-dlp for %s", url)
        return [], []

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _extract_frames_cv2(
        self, video_path: str
    ) -> tuple[list[float], list[np.ndarray]]:
        """使用 OpenCV 按间隔抽帧。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_sec = total_frames / fps

        skip_frames = int(self.skip_head_sec * fps)
        step_frames = max(1, int(self.sample_every_sec * fps))

        timestamps: list[float] = []
        frames: list[np.ndarray] = []

        frame_idx = skip_frames
        while frame_idx < total_frames:
            if len(frames) >= self.max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            timestamps.append(frame_idx / fps)
            frames.append(frame)
            frame_idx += step_frames

        cap.release()
        logger.info(
            "Sampled %d frames from '%s' (total=%.1fs, skip=%.1fs, every=%.1fs)",
            len(frames),
            video_path,
            total_sec,
            self.skip_head_sec,
            self.sample_every_sec,
        )
        return timestamps, frames
