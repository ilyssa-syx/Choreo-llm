"""
downloader.py
=============
用 yt-dlp 将筛选后的 Bilibili 视频下载到本地 .mp4 文件。

特性：
  - 支持多线程并发下载（ThreadPoolExecutor）
  - 自动跳过已下载的文件
  - 支持限速、cookies 文件（用于已登录账号，降低被限制风险）
  - 下载到 {download_dir}/{label}/{bvid}.mp4
  - final_recommendation 不在下载列表中的视频直接跳过

使用前请确认 yt-dlp 已安装：
  pip install yt-dlp
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import PipelineConfig, DEFAULT_CONFIG
from .models import VideoRecord

logger = logging.getLogger(__name__)

# Bilibili 视频页 URL 前缀
_BILI_VIDEO_PREFIX = "https://www.bilibili.com/video/"


def _bvid_to_url(bvid: str) -> str:
    return f"{_BILI_VIDEO_PREFIX}{bvid}"


class VideoDownloader:
    """使用 yt-dlp 下载 Bilibili 视频到本地。

    Args:
        config: 全局 PipelineConfig 实例
        cookies_file: yt-dlp cookies 文件路径（可覆盖 config.download_cookies_file）
    """

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        cookies_file: Optional[str] = None,
    ) -> None:
        self._cfg = config
        self._cookies_file = cookies_file or config.download_cookies_file or None

        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp 未找到。请先安装：pip install yt-dlp\n"
                "安装后重新运行 pipeline。"
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def download_batch(
        self,
        records: list[VideoRecord],
        force_all_candidates: bool = False,
    ) -> tuple[list[VideoRecord], list[VideoRecord]]:
        """批量下载，返回 (成功列表, 失败列表)。

        会根据 config 自动决定下载哪些 label / score 的视频。
        已有 downloaded_path 且文件存在的视频直接跳过。

        Args:
            records: 视频记录列表
            force_all_candidates: 若为 True，则跳过 label/score 过滤，
                                  下载所有候选视频（用于 vision/audio 检测前）
        """
        to_download = self._select_records(records, force_all_candidates=force_all_candidates)
        if not to_download:
            logger.info("没有符合下载条件的视频（label/score 过滤后为空）。")
            return [], []

        logger.info(
            "准备下载 %d 个视频（workers=%d）→ %s",
            len(to_download),
            self._cfg.download_workers,
            self._cfg.download_dir,
        )

        succeeded: list[VideoRecord] = []
        failed: list[VideoRecord] = []

        # 候选阶段下载：强制使用 candidates 目录
        force_candidates = force_all_candidates

        with ThreadPoolExecutor(max_workers=self._cfg.download_workers) as pool:
            futures = {
                pool.submit(self._download_one, rec, force_candidates): rec
                for rec in to_download
            }
            with tqdm(total=len(futures), desc="Downloading", unit="video") as pbar:
                for future in as_completed(futures):
                    rec = futures[future]
                    try:
                        result = future.result()
                        if result:
                            succeeded.append(rec)
                            pbar.set_postfix(ok=len(succeeded), fail=len(failed))
                        else:
                            failed.append(rec)
                    except Exception as exc:
                        logger.error("下载异常 bvid=%s: %s", rec.bvid, exc)
                        failed.append(rec)
                    finally:
                        pbar.update(1)

        total_size_gb = self._calc_total_size_gb(succeeded)
        logger.info(
            "下载完成: 成功 %d / 失败 %d，合计 %.2f GB",
            len(succeeded),
            len(failed),
            total_size_gb,
        )
        if failed:
            logger.warning(
                "失败视频（bvid）: %s",
                ", ".join(r.bvid for r in failed[:20]),
            )
        return succeeded, failed

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _select_records(
        self,
        records: list[VideoRecord],
        force_all_candidates: bool = False,
    ) -> list[VideoRecord]:
        """根据配置决定哪些视频需要下载。

        Args:
            records: 视频记录列表
            force_all_candidates: 若为 True，跳过 label/score 过滤
        """
        selected = []
        for rec in records:
            # 候选阶段：无条件下载所有视频（除了已有本地文件的）
            if force_all_candidates:
                # 已有本地文件则跳过
                if rec.downloaded_path and Path(rec.downloaded_path).exists():
                    logger.debug("已存在，跳过: %s", rec.downloaded_path)
                    continue
                selected.append(rec)
                continue

            # 正常下载阶段：根据 label/score 过滤
            # label 过滤
            if rec.final_recommendation == "keep" and not self._cfg.download_keep:
                continue
            if rec.final_recommendation == "review" and not self._cfg.download_review:
                continue
            if rec.final_recommendation == "drop":
                continue
            # score 过滤
            if (
                self._cfg.download_min_fusion_score > 0
                and rec.final_score < self._cfg.download_min_fusion_score
            ):
                continue
            # 已有本地文件则跳过
            if rec.downloaded_path and Path(rec.downloaded_path).exists():
                logger.debug("已存在，跳过: %s", rec.downloaded_path)
                continue
            selected.append(rec)
        return selected

    def _download_one(self, rec: VideoRecord, force_candidates_dir: bool = False) -> bool:
        """下载单个视频，成功后更新 rec.downloaded_path，返回是否成功。
        
        Args:
            rec: 视频记录
            force_candidates_dir: 若为 True，强制下载到 candidates/ 目录
        """
        # 候选阶段强制下载到 candidates/，评分后下载到对应 label 目录
        if force_candidates_dir:
            subdir = "candidates"
        else:
            subdir = rec.final_recommendation
        
        label_dir = Path(self._cfg.download_dir) / subdir
        label_dir.mkdir(parents=True, exist_ok=True)
        out_path = label_dir / f"{rec.bvid}.mp4"

        # 若文件已存在（别的线程刚下好），直接标记
        if out_path.exists():
            rec.downloaded_path = str(out_path)
            return True

        url = rec.url or _bvid_to_url(rec.bvid)
        cmd = self._build_cmd(url, out_path)

        logger.debug("yt-dlp cmd: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 最长等 10 分钟
            )
        except subprocess.TimeoutExpired:
            logger.error("下载超时 bvid=%s", rec.bvid)
            return False
        except Exception as exc:
            logger.error("下载异常 bvid=%s: %s", rec.bvid, exc)
            return False

        if result.returncode != 0:
            logger.warning(
                "yt-dlp 失败 bvid=%s (code=%d):\n%s",
                rec.bvid,
                result.returncode,
                result.stderr[-500:],  # 只取最后 500 字符避免日志过长
            )
            return False

        if out_path.exists():
            rec.downloaded_path = str(out_path)
            logger.debug("下载成功: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
            return True
        else:
            # yt-dlp 可能将文件写到其他格式，尝试找一下
            candidates = list(label_dir.glob(f"{rec.bvid}.*"))
            if candidates:
                rec.downloaded_path = str(candidates[0])
                logger.debug("下载成功（非标准扩展名）: %s", candidates[0])
                return True
            logger.warning("yt-dlp 退出码 0 但未找到文件: bvid=%s", rec.bvid)
            return False

    def _build_cmd(self, url: str, out_path: Path) -> list[str]:
        """构建 yt-dlp 命令行参数列表。"""
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--format", self._cfg.download_format,
            "--output", str(out_path),
            "--no-warnings",
            "--retries", "3",
            "--fragment-retries", "3",
            "--merge-output-format", "mp4",
        ]
        if self._cookies_file and Path(self._cookies_file).exists():
            cmd += ["--cookies", self._cookies_file]
        if self._cfg.download_limit_rate:
            cmd += ["--limit-rate", self._cfg.download_limit_rate]
        # 设置帧率（使用 ffmpeg 后处理）
        if self._cfg.download_fps > 0:
            cmd += ["--postprocessor-args", f"ffmpeg:-r {self._cfg.download_fps}"]
        cmd.append(url)
        return cmd

    @staticmethod
    def _calc_total_size_gb(records: list[VideoRecord]) -> float:
        total = 0.0
        for rec in records:
            if rec.downloaded_path:
                p = Path(rec.downloaded_path)
                if p.exists():
                    total += p.stat().st_size
        return total / 1e9
