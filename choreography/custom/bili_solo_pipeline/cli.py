"""
cli.py
======
命令行入口，使用 argparse 解析参数并驱动 Pipeline。

使用示例：

  # 基本运行（使用内置舞蹈关键词）
  python -m bili_solo_pipeline.cli \\
    --out_dir data/bili_solo_candidates \\
    --max_pages_per_query 5

  # 自定义关键词文件
  python -m bili_solo_pipeline.cli \\
    --queries_file keywords.txt \\
    --out_dir data/bili_solo_candidates \\
    --min_duration_sec 30

  # 仅从 dedup 阶段开始重跑（已有缓存）
  python -m bili_solo_pipeline.cli \\
    --out_dir data/bili_solo_candidates \\
    --run_from_stage dedup

  # 启用音频评分，禁用姿态估计
  python -m bili_solo_pipeline.cli \\
    --out_dir data/bili_solo_candidates \\
    --enable_audio_scoring true \\
    --enable_pose false \\
    --solo_threshold 0.75 \\
    --audio_threshold 0.60

  # 提供本地视频目录用于视觉/音频检测
  python -m bili_solo_pipeline.cli \\
    --out_dir data/bili_solo_candidates \\
    --video_dir /data/bili_videos
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import PipelineConfig, DEFAULT_CONFIG
from .pipeline import Pipeline, STAGES
from .utils import setup_logging


def _bool_arg(value: str) -> bool:
    """argparse 布尔值解析（支持 true/false/1/0/yes/no）。"""
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bili-solo-pipeline",
        description="Bilibili 单人独舞视频候选集构建工具（仅用于研究候选集整理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- 搜索参数 ----
    search_grp = parser.add_argument_group("搜索参数")
    search_grp.add_argument(
        "--queries_file", type=str, default=None,
        metavar="PATH",
        help="关键词文件路径（每行一个词，'#' 开头为注释）。不提供则使用内置舞种词表。",
    )
    search_grp.add_argument(
        "--max_pages_per_query", type=int, default=None,
        metavar="N",
        help=f"每个关键词最多搜索页数（默认 {DEFAULT_CONFIG.max_pages_per_query}）",
    )
    search_grp.add_argument(
        "--search_order", type=str, default=None,
        choices=["totalrank", "click", "pubdate", "dm", "stow"],
        help="排序方式（默认 totalrank）",
    )
    search_grp.add_argument(
        "--request_delay_sec", type=float, default=None,
        metavar="SEC",
        help=f"每次请求最小间隔秒数（默认 {DEFAULT_CONFIG.request_delay_sec}，建议 >= 1.0）",
    )
    search_grp.add_argument(
        "--cookies_file", type=str, default=None,
        metavar="PATH",
        help="B 站 Cookie JSON 文件（可选，提供后搜索更稳定）",
    )
    search_grp.add_argument(
        "--target_hours", type=float, default=None,
        metavar="H",
        dest="target_total_hours",
        help="搜索目标总时长（小时）；累计达到后自动停止，默认 0 = 不限制。示例：--target_hours 10",
    )
    search_grp.add_argument(
        "--search_oversample_rate", type=float, default=None,
        metavar="N",
        help=(
            f"原始素材超采样倍率，默认 {DEFAULT_CONFIG.search_oversample_rate}。"
            "搜索 target×N 小时的原始视频来弥补过滤损耗。"
            "过滤严（如硬舞种限定）可调高到 6~8。"
        ),
    )

    # ---- 元数据过滤 ----
    filter_grp = parser.add_argument_group("元数据过滤")
    filter_grp.add_argument(
        "--min_duration_sec", type=float, default=None,
        metavar="SEC",
        help=f"最短视频时长（默认 {DEFAULT_CONFIG.min_duration_sec}s）",
    )
    filter_grp.add_argument(
        "--max_duration_sec", type=float, default=None,
        metavar="SEC",
        help=f"最长视频时长（默认 {DEFAULT_CONFIG.max_duration_sec}s）",
    )
    filter_grp.add_argument(
        "--min_view_count", type=int, default=None,
        metavar="N",
        help="最低播放量（默认 0，不限制）",
    )

    # ---- 视觉检测 ----
    vision_grp = parser.add_argument_group("视觉检测")
    vision_grp.add_argument(
        "--video_dir", type=str, default=None,
        metavar="DIR",
        help="本地视频目录（文件名格式：{bvid}.mp4）。不提供则跳过视觉/音频检测。",
    )
    vision_grp.add_argument(
        "--sample_every_sec", type=float, default=None,
        metavar="SEC",
        help=f"抽帧间隔（默认 {DEFAULT_CONFIG.sample_every_sec}s）",
    )
    vision_grp.add_argument(
        "--max_frames", type=int, default=None,
        metavar="N",
        help=f"最大抽帧数（默认 {DEFAULT_CONFIG.max_frames}）",
    )
    vision_grp.add_argument(
        "--yolo_model", type=str, default=None,
        metavar="PATH",
        help=f"YOLO 模型权重（默认 {DEFAULT_CONFIG.yolo_model}）",
    )
    vision_grp.add_argument(
        "--enable_pose", type=_bool_arg, default=None,
        metavar="BOOL",
        help="是否启用姿态估计（默认 false）",
    )

    # ---- 单人评分 ----
    solo_grp = parser.add_argument_group("单人评分")
    solo_grp.add_argument(
        "--solo_threshold", type=float, default=None,
        metavar="FLOAT",
        help=f"视觉 solo_score >= 此值视为 likely_solo（默认 {DEFAULT_CONFIG.solo_threshold}）",
    )
    solo_grp.add_argument(
        "--uncertain_threshold", type=float, default=None,
        metavar="FLOAT",
        help=f"视觉 score >= 此值且 < solo_threshold 视为 uncertain（默认 {DEFAULT_CONFIG.uncertain_threshold}）",
    )

    # ---- 音频评分 ----
    audio_grp = parser.add_argument_group("音频评分")
    audio_grp.add_argument(
        "--enable_audio_scoring", type=_bool_arg, default=None,
        metavar="BOOL",
        help=f"是否启用音频评分（默认 {DEFAULT_CONFIG.enable_audio_scoring}）",
    )
    audio_grp.add_argument(
        "--audio_threshold", type=float, default=None,
        metavar="FLOAT",
        help=f"audio_score >= 此值视为 clear_music（默认 {DEFAULT_CONFIG.audio_threshold}）",
    )
    audio_grp.add_argument(
        "--speech_ratio_max", type=float, default=None,
        metavar="FLOAT",
        help=f"语音占比超过此值进入 review（默认 {DEFAULT_CONFIG.speech_ratio_max}）",
    )

    # ---- 去重 ----
    dedup_grp = parser.add_argument_group("去重")
    dedup_grp.add_argument(
        "--enable_phash_dedup", type=_bool_arg, default=None,
        metavar="BOOL",
        help="是否启用封面感知哈希去重（默认 false，需 imagehash+Pillow）",
    )

    # ---- 输出 ----
    out_grp = parser.add_argument_group("输出")
    out_grp.add_argument(
        "--out_dir", type=str, default=DEFAULT_CONFIG.out_dir,
        metavar="DIR",
        help=f"输出目录（默认 {DEFAULT_CONFIG.out_dir}）",
    )
    out_grp.add_argument(
        "--use_cache", type=_bool_arg, default=None,
        metavar="BOOL",
        help="是否使用断点续跑缓存（默认 true）",
    )

    # ---- 视频下载 ----
    dl_grp = parser.add_argument_group("视频下载（需要 yt-dlp）")
    dl_grp.add_argument(
        "--download", dest="enable_download",
        action="store_true", default=False,
        help="启用视频下载（pipeline 末自动下载筛选后视频）",
    )
    dl_grp.add_argument(
        "--download_dir", type=str, default=None,
        metavar="DIR",
        help=f"视频下载根目录（默认 {DEFAULT_CONFIG.download_dir}）",
    )
    dl_grp.add_argument(
        "--download_review", type=_bool_arg, default=None,
        metavar="BOOL",
        help="是否也下载 review 标签的视频（默认 false，仅下载 keep）",
    )
    dl_grp.add_argument(
        "--download_workers", type=int, default=None,
        metavar="N",
        help=f"并发下载线程数（默认 {DEFAULT_CONFIG.download_workers}）",
    )
    dl_grp.add_argument(
        "--download_min_score", type=float, default=None,
        dest="download_min_fusion_score",
        metavar="FLOAT",
        help="仅下载 final_score >= 此值的视频（默认 0 = 不限）",
    )
    dl_grp.add_argument(
        "--download_limit_rate", type=str, default=None,
        metavar="RATE",
        help="yt-dlp 限速，如 '5M' 限 5MB/s（默认不限）",
    )
    dl_grp.add_argument(
        "--download_cookies_file", type=str, default=None,
        metavar="PATH",
        help="提供给 yt-dlp 的 cookies 文件（Netscape 格式，可降低下载被限风险）",
    )

    # ---- 流程控制 ----
    ctrl_grp = parser.add_argument_group("流程控制")
    ctrl_grp.add_argument(
        "--run_from_stage", type=str, default="search",
        choices=STAGES,
        help=f"从哪个阶段开始运行（前面阶段从缓存加载），可选: {STAGES}",
    )

    # ---- 日志 ----
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        metavar="PATH",
        help="日志文件路径（可选，默认只输出到控制台）",
    )

    # ---- 高级：JSON 覆盖 ----
    parser.add_argument(
        "--config_override", type=str, default=None,
        metavar="JSON",
        help="JSON 字符串，用于覆盖任意配置字段。例如: '{\"weight_solo\": 0.6}'",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 初始化日志
    setup_logging(
        level=args.log_level,
        log_file=args.log_file or str(Path(args.out_dir) / "pipeline.log"),
    )
    logger = logging.getLogger(__name__)
    logger.info("bili-solo-pipeline starting. Args: %s", vars(args))

    # 构建配置（从 DEFAULT 开始，逐步覆盖）
    import dataclasses
    cfg_fields = {f.name: getattr(DEFAULT_CONFIG, f.name)
                  for f in dataclasses.fields(DEFAULT_CONFIG)}
    cfg_fields["out_dir"] = args.out_dir

    # 逐个 CLI 参数覆盖
    override_map = {
        "queries_file": args.queries_file,
        "max_pages_per_query": args.max_pages_per_query,
        "search_order": args.search_order,
        "request_delay_sec": args.request_delay_sec,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "min_view_count": args.min_view_count,
        "sample_every_sec": args.sample_every_sec,
        "max_frames": args.max_frames,
        "yolo_model": args.yolo_model,
        "enable_pose": args.enable_pose,
        "solo_threshold": args.solo_threshold,
        "uncertain_threshold": args.uncertain_threshold,
        "enable_audio_scoring": args.enable_audio_scoring,
        "audio_threshold": args.audio_threshold,
        "speech_ratio_max": args.speech_ratio_max,
        "enable_phash_dedup": args.enable_phash_dedup,
        "use_cache": args.use_cache,
        "log_level": args.log_level,
        # 下载参数（--download 是 store_true，仅显式传时覆盖）
        "enable_download": True if args.enable_download else None,
        "download_dir": args.download_dir,
        "download_review": args.download_review,
        "download_workers": args.download_workers,
        "download_min_fusion_score": args.download_min_fusion_score,
        "download_limit_rate": args.download_limit_rate,
        "download_cookies_file": args.download_cookies_file,
        # target_hours / oversample（之前加的）
        "target_total_hours": getattr(args, "target_total_hours", None),
        "search_oversample_rate": getattr(args, "search_oversample_rate", None),
    }
    for key, value in override_map.items():
        if value is not None:
            cfg_fields[key] = value

    # JSON 覆盖（优先级最高）
    if args.config_override:
        try:
            overrides = json.loads(args.config_override)
            for k, v in overrides.items():
                if k in cfg_fields:
                    cfg_fields[k] = v
                    logger.info("Config override: %s = %s", k, v)
                else:
                    logger.warning("Unknown config key in --config_override: %s", k)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse --config_override JSON: %s", exc)
            sys.exit(1)

    config = PipelineConfig(**cfg_fields)

    # 输出配置摘要
    logger.info(
        "Config summary: min_dur=%.0fs max_pages=%d solo_thr=%.2f audio_thr=%.2f "
        "vision_enabled=%s audio_enabled=%s",
        config.min_duration_sec,
        config.max_pages_per_query,
        config.solo_threshold,
        config.audio_threshold,
        args.video_dir is not None,
        config.enable_audio_scoring,
    )

    # 运行 Pipeline
    pipeline = Pipeline(
        config=config,
        cookies_file=args.cookies_file,
        run_from_stage=args.run_from_stage,
        video_dir=args.video_dir,
    )
    try:
        records = pipeline.run()
        logger.info("Pipeline completed. %d candidates processed.", len(records))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Partial results may be saved in cache.")
        sys.exit(0)
    except Exception as exc:
        logger.error("Pipeline failed with unexpected error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
