"""
search_client.py
================
Bilibili 搜索客户端。

使用 B 站公开搜索 API（search.bilibili.com）。
注意：
  1. 不绕过任何验证码/风控机制
  2. 所有请求带速率限制与指数退避重试
  3. 若接口返回认证失败（-101 / -400 等），优雅降级并提示人工处理
  4. 该接口为公开可访问接口，但 B 站可能随时修改；若失效请更新 _API_URL

接口说明（公开可查文档）：
  https://github.com/SocialSisterYi/bilibili-API-collect （社区整理的非官方文档）
  仅使用无需登录的搜索端点：/x/web-interface/search/type

WBI 签名（2023 年 B 站新增反爬机制）：
  B 站 API 要求在请求参数中附加 w_rid（MD5 签名）和 wts（时间戳），
  签名密钥从 /x/web-interface/nav 动态获取，详见 _WbiSigner 类。
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import random
import re
import time
import urllib.parse
from functools import lru_cache
from typing import Iterator, Optional

import requests

from .config import PipelineConfig, DEFAULT_CONFIG
from .models import RawSearchItem
from .utils import RateLimiter, retry_with_backoff, parse_duration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# B站2022年后搜索API必须用 /wbi/ 前缀的端点（旧端点已严格限制）
_API_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
# 预热 URL：第一次请求前 GET 此地址以获取 buvid3 等必要 Cookie
_WARM_UP_URL = "https://www.bilibili.com"

# B 站搜索分区 ID：舞蹈（tid=20）
# 若不限分区可不传 tids 参数
_DANCE_TID = 20

# 公开 User-Agent（模拟普通浏览器访问）
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Origin": "https://www.bilibili.com",
}

# WBI 混淆表（固定不变，来源：社区文档）
_MIXIN_KEY_ENC_TAB: list[int] = [
    46, 47, 18,  2, 53,  8, 23, 32, 15, 50, 10, 31, 58,  3, 45, 35,
    27, 43,  5, 49, 33,  9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48,  7, 16, 24, 55, 40, 61, 26, 17,  0,  1, 60, 51, 30,  4,
    22, 25, 54, 21, 56, 59,  6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

# 已知的 B 站错误码及说明
_BILI_ERROR_CODES: dict[int, str] = {
    -101: "用户未登录（certain endpoints require login）",
    -400: "请求错误（参数有误）",
    -403: "访问权限不足",
    -404: "资源不存在",
    -412: "请求被拦截（WBI 签名无效或触发风控，建议增加延迟或提供 cookies）",
    -509: "请求频率过高",
    0: "SUCCESS",
}

# WBI 密钥缓存有效期（秒）—— B 站约每天轮换一次，保险起见缓存 12 小时
_WBI_KEY_TTL = 43200


# ---------------------------------------------------------------------------
# WBI 签名器
# ---------------------------------------------------------------------------

class _WbiSigner:
    """为 Bilibili API 请求添加 WBI 签名（w_rid + wts）。

    B 站于 2023 年引入此机制：对请求参数按字母序排列后拼接动态 mixin_key，
    再做 MD5 得到 w_rid。密钥通过 /x/web-interface/nav 端点动态获取。

    参考：https://github.com/SocialSisterYi/bilibili-API-collect/blob/master/docs/misc/sign/wbi.md
    """

    def __init__(self, session: requests.Session) -> None:
        self._session = session
        self._mixin_key: Optional[str] = None
        self._key_fetch_time: float = 0.0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def sign(self, params: dict) -> dict:
        """在 params 副本中注入 wts 和 w_rid，返回新字典。"""
        mixin_key = self._get_mixin_key()
        wts = int(time.time())
        signed = dict(params)
        signed["wts"] = wts
        # 过滤掉参数值中的 '!' '"' "'" '(' ')' '*' 等特殊字符（B 站要求）
        filtered = {
            k: re.sub(r"[!'\"()*]", "", str(v))
            for k, v in sorted(signed.items())
        }
        query_str = urllib.parse.urlencode(filtered) + mixin_key
        w_rid = hashlib.md5(query_str.encode()).hexdigest()
        filtered["w_rid"] = w_rid
        return filtered

    def invalidate(self) -> None:
        """强制下次调用时重新获取 mixin_key（例如收到 412 后调用）。"""
        self._mixin_key = None
        self._key_fetch_time = 0.0

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _get_mixin_key(self) -> str:
        """返回有效的 mixin_key，必要时重新获取。"""
        if self._mixin_key and (time.time() - self._key_fetch_time) < _WBI_KEY_TTL:
            return self._mixin_key
        self._mixin_key = self._fetch_mixin_key()
        self._key_fetch_time = time.time()
        return self._mixin_key

    def _fetch_mixin_key(self) -> str:
        """从 /x/web-interface/nav 获取并推导 mixin_key。"""
        try:
            resp = self._session.get(_NAV_URL, timeout=10)
            resp.raise_for_status()
            nav = resp.json()
        except Exception as exc:
            logger.warning("WBI: 无法获取 nav 端点，WBI 签名将被跳过: %s", exc)
            return ""

        try:
            wbi_img = nav["data"]["wbi_img"]
            img_key = wbi_img["img_url"].rsplit("/", 1)[-1].split(".")[0]
            sub_key = wbi_img["sub_url"].rsplit("/", 1)[-1].split(".")[0]
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("WBI: 解析 wbi_img 字段失败: %s", exc)
            return ""

        raw_key = img_key + sub_key
        # 按混淆表重排，取前 32 字符
        mixin_key = "".join(
            raw_key[idx] for idx in _MIXIN_KEY_ENC_TAB if idx < len(raw_key)
        )[:32]
        logger.debug("WBI mixin_key 已刷新（前8字符: %s...）", mixin_key[:8])
        return mixin_key


# ---------------------------------------------------------------------------
# 搜索客户端
# ---------------------------------------------------------------------------

class BiliSearchClient:
    """Bilibili 视频搜索客户端。

    Args:
        config: 全局 PipelineConfig 实例
        cookies: 可选 cookies 字典（若账号登录可提供，非必须）
    """

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        cookies: Optional[dict[str, str]] = None,
    ) -> None:
        self._cfg = config
        self._session = requests.Session()
        self._session.headers.update(_DEFAULT_HEADERS)
        if cookies:
            self._session.cookies.update(cookies)
        self._limiter = RateLimiter(min_interval=config.request_delay_sec)
        self._wbi = _WbiSigner(self._session)
        # 文档要求：搜索前先 GET bilibili.com 以获取 buvid3 等初始 Cookie
        self._warm_up_session()

    # ------------------------------------------------------------------
    # 预热 / Session 初始化
    # ------------------------------------------------------------------

    def _warm_up_session(self) -> None:
        """GET bilibili.com 主页以获取必要的初始 Cookie（buvid3 等）。

        根据官方文档：B 站于 2022 年 8 月要求搜索 API 携带 Cookie，
        若缺少 buvid3 等字段将返回 -412。通过访问主页可自动获取。
        """
        try:
            resp = self._session.get(_WARM_UP_URL, timeout=10)
            resp.raise_for_status()
            logger.debug(
                "Session 预热完成，获取到 Cookie: %s",
                list(self._session.cookies.keys()),
            )
        except Exception as exc:
            logger.warning("Session 预热失败（将继续尝试）: %s", exc)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def search_query(
        self,
        query: str,
        order: Optional[str] = None,
        max_pages: Optional[int] = None,
        tid: Optional[int] = _DANCE_TID,
    ) -> Iterator[RawSearchItem]:
        """搜索单个关键词，分页迭代返回结果。

        Args:
            query: 搜索词
            order: 排序方式（totalrank/click/pubdate/dm/stow）；None 使用配置默认值
            max_pages: 最大页数；None 使用配置默认值
            tid: 限定分区 ID（默认 20=舞蹈）；传 None 不限分区

        Yields:
            RawSearchItem 实例
        """
        order = order or self._cfg.search_order
        max_pages = max_pages or self._cfg.max_pages_per_query

        logger.info("Searching query='%s', order=%s, max_pages=%d", query, order, max_pages)

        for page in range(1, max_pages + 1):
            items = self._fetch_page(query, page=page, order=order, tid=tid)
            if items is None:
                # 发生不可恢复错误，停止该关键词的搜索
                logger.warning("Stopping search for query='%s' at page %d due to error.", query, page)
                return
            if not items:
                logger.info("No more results for query='%s' at page %d.", query, page)
                return
            for item in items:
                yield item

    def search_many_queries(
        self,
        queries: list[str],
        order: Optional[str] = None,
        max_pages: Optional[int] = None,
        tid: Optional[int] = _DANCE_TID,
    ) -> Iterator[tuple[str, RawSearchItem]]:
        """批量搜索多个关键词。

        Yields:
            (query, RawSearchItem) 元组，方便追踪命中关键词
        """
        for query in queries:
            for item in self.search_query(query, order=order, max_pages=max_pages, tid=tid):
                item.search_query = query
                yield query, item

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    @retry_with_backoff(max_retries=4, base_delay=5.0)
    def _raw_get(self, params: dict) -> dict:
        """执行带重试的 GET 请求，返回 JSON 响应。

        自动附加 WBI 签名（w_rid + wts）；若收到 HTTP 412 则：
          1. 刷新 WBI 密钥（防止密钥过期导致的 412）
          2. 额外等待 15–30 秒冷却（IP 级限速通常需要更长冷却窗口）
          3. 抛出异常由 retry_with_backoff 装饰器继续退避重试
        """
        self._limiter.wait()
        signed_params = self._wbi.sign(params)
        resp = self._session.get(
            _API_URL,
            params=signed_params,
            timeout=self._cfg.search_timeout_sec,
        )
        if resp.status_code == 412:
            cooldown = random.uniform(15, 30)
            logger.warning(
                "收到 HTTP 412（IP 级限速），刷新 WBI 密钥并冷却 %.0f 秒后重试…",
                cooldown,
            )
            self._wbi.invalidate()
            time.sleep(cooldown)
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    def _fetch_page(
        self,
        query: str,
        page: int,
        order: str,
        tid: Optional[int],
    ) -> Optional[list[RawSearchItem]]:
        """拉取单页搜索结果。

        Returns:
            RawSearchItem 列表；若接口错误返回 None；若无结果返回空列表
        """
        params: dict = {
            "search_type": "video",
            "keyword": query,
            "page": page,
            "page_size": self._cfg.results_per_page,
            "order": order,
        }
        if tid is not None:
            params["tids"] = tid

        try:
            data = self._raw_get(params)
        except requests.RequestException as exc:
            logger.error("HTTP error fetching page %d for query='%s': %s", page, query, exc)
            return None

        # 检查 B 站业务错误码
        code = data.get("code", 0)
        if code != 0:
            msg = _BILI_ERROR_CODES.get(code, f"未知错误码 {code}")
            if code in (-412, -509):
                # 风控/限速，建议人工处理
                logger.error(
                    "Bilibili rate-limit/anti-bot triggered (code=%d: %s). "
                    "Consider increasing request_delay_sec or adding cookies. "
                    "Manual intervention may be required.",
                    code,
                    msg,
                )
            elif code == -101:
                logger.warning(
                    "Bilibili API requires login (code=%d). "
                    "Provide --cookies_file for better results.",
                    code,
                )
            else:
                logger.warning("Bilibili API error code=%d: %s", code, msg)
            return None

        result_data = data.get("data", {})
        raw_list: list[dict] = result_data.get("result", []) or []

        if not raw_list:
            return []

        items: list[RawSearchItem] = []
        for raw in raw_list:
            item = self._parse_raw_item(raw, query)
            if item is not None:
                items.append(item)

        logger.debug("Page %d for '%s': fetched %d items", page, query, len(items))
        return items

    @staticmethod
    def _parse_raw_item(raw: dict, query: str) -> Optional[RawSearchItem]:
        """将 API 原始字典解析为 RawSearchItem。"""
        try:
            bvid = raw.get("bvid", "")
            aid = str(raw.get("aid", ""))
            title = raw.get("title", "")
            # B 站搜索结果标题带 <em> 高亮标签，去除之
            import re
            title = re.sub(r"<[^>]+>", "", title)

            # 时长字符串 -> 秒（B 站格式 "MM:SS" 或 "HH:MM:SS"）
            duration_str = str(raw.get("duration", "0:00"))
            duration_sec = parse_duration(duration_str)

            # 发布时间戳
            pubdate = raw.get("pubdate", 0)
            publish_ts = int(pubdate)

            # UP 主
            author = raw.get("author", "")

            # 封面
            pic = raw.get("pic", "")
            if pic and not pic.startswith("http"):
                pic = "https:" + pic

            # 播放量（B 站搜索结果里的字段名为 play）
            play = raw.get("play", 0)
            if isinstance(play, str):
                play = play.replace(",", "")
                try:
                    play = int(play)
                except ValueError:
                    play = 0

            # 分区
            tid = raw.get("typeid", 0)
            try:
                tid = int(tid)
            except (ValueError, TypeError):
                tid = 0
            tid_name = raw.get("typename", "")

            # 标签（搜索结果通常没有 tag 字段，需要后续单独获取详情）
            tag = raw.get("tag", "")
            tags = [t.strip() for t in tag.split(",") if t.strip()] if tag else []

            return RawSearchItem(
                bvid=bvid,
                aid=aid,
                title=title,
                uploader=author,
                duration_str=duration_str,
                duration_sec=duration_sec,
                publish_ts=publish_ts,
                view_count=play,
                cover_url=pic,
                tid=tid,
                tid_name=tid_name,
                tags=tags,
                search_query=query,
            )
        except Exception as exc:
            logger.warning("Failed to parse search item: %s | raw=%s", exc, raw)
            return None


def raw_item_to_video_url(bvid: str) -> str:
    """根据 bvid 生成视频页 URL。"""
    return f"https://www.bilibili.com/video/{bvid}"
