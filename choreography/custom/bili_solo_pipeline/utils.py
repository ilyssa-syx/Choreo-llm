"""
utils.py
========
公共工具：日志初始化、速率限制器、指数退避重试、时长解析、文本归一化等。
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
import unicodedata
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import requests

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# 日志初始化
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """统一配置 root logger，支持同时输出到控制台和文件。"""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )
    # 降低 urllib3/requests 的噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 速率限制器（令牌桶 / 简单最小间隔实现）
# ---------------------------------------------------------------------------

class RateLimiter:
    """简单的最小间隔速率限制器。

    用法::

        limiter = RateLimiter(min_interval=1.5)
        for url in urls:
            limiter.wait()
            resp = requests.get(url)
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = min_interval
        self._last_call: float = 0.0

    def wait(self) -> None:
        """阻塞直到满足最小间隔要求。"""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            sleep_for = self._min_interval - elapsed
            logger.debug("RateLimiter: sleeping %.2f s", sleep_for)
            time.sleep(sleep_for)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# 指数退避重试装饰器
# ---------------------------------------------------------------------------

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
    ),
) -> Callable[[F], F]:
    """装饰器：对指定异常类型进行指数退避重试。

    Args:
        max_retries: 最大重试次数（不含第一次调用）
        base_delay: 退避基础延迟（秒），实际延迟 = base_delay * 2^i + jitter
        exceptions: 触发重试的异常类型元组

    Example::

        @retry_with_backoff(max_retries=3)
        def fetch(url: str) -> dict: ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            "Function %s failed after %d retries: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
                        raise
                    # 指数退避 + 随机抖动，避免被识别为规律性请求
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, base_delay)
                    logger.warning(
                        "Function %s attempt %d/%d failed: %s. Retrying in %.1f s...",
                        func.__name__,
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise RuntimeError("Unreachable") from last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# 时长解析工具
# ---------------------------------------------------------------------------

def parse_duration(raw: str) -> float:
    """将 B 站时长字符串（如 '3:25' / '1:02:35'）解析为秒数。

    Args:
        raw: 原始时长字符串

    Returns:
        float: 秒数；解析失败返回 0.0
    """
    raw = raw.strip()
    if not raw:
        return 0.0
    parts = raw.split(":")
    try:
        parts_int = [int(p) for p in parts]
    except ValueError:
        logger.debug("parse_duration: cannot parse '%s'", raw)
        return 0.0

    if len(parts_int) == 2:
        return parts_int[0] * 60 + parts_int[1]
    elif len(parts_int) == 3:
        return parts_int[0] * 3600 + parts_int[1] * 60 + parts_int[2]
    else:
        return float(parts_int[-1])


# ---------------------------------------------------------------------------
# 文本归一化（用于标题去重 & 关键词匹配）
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """将标题归一化：繁简转换（需 opencc，可选）、全角->半角、去标点等。

    若 opencc 未安装，跳过繁简转换并记录 debug 日志。
    """
    # 全角转半角
    result = unicodedata.normalize("NFKC", title)
    # 去除多余空白
    result = re.sub(r"\s+", " ", result).strip()
    # 小写
    result = result.lower()
    return result


def contains_any(text: str, keywords: list[str]) -> list[str]:
    """返回 text 中命中的关键词列表（大小写不敏感）。"""
    text_low = text.lower()
    return [kw for kw in keywords if kw.lower() in text_low]


# ---------------------------------------------------------------------------
# 文件 / 缓存工具
# ---------------------------------------------------------------------------

def md5_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    import shutil
    available = shutil.which("ffmpeg") is not None
    if not available:
        logger.warning("ffmpeg not found in PATH; audio extraction will be skipped.")
    return available


def check_yt_dlp() -> bool:
    """检查 yt-dlp 是否可用（用于可选视频流抓取）。"""
    import shutil
    available = shutil.which("yt-dlp") is not None
    if not available:
        logger.info("yt-dlp not found; video stream download will not be available. "
                    "Provide local video paths manually for vision/audio scoring.")
    return available
