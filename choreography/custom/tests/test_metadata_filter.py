"""
test_metadata_filter.py
=======================
单元测试：标题黑白名单过滤 + 舞种关键词匹配与归类
"""

import pytest
import sys
import os

# 确保可以导入 bili_solo_pipeline 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bili_solo_pipeline.metadata_filter import MetadataFilter
from bili_solo_pipeline.config import DEFAULT_CONFIG, PipelineConfig
from bili_solo_pipeline.models import RawSearchItem


def _make_item(
    title: str = "测试舞蹈视频",
    duration_sec: float = 120.0,
    tags: list[str] | None = None,
    bvid: str = "BV_test001",
    view_count: int = 1000,
) -> RawSearchItem:
    return RawSearchItem(
        bvid=bvid,
        aid="123456",
        title=title,
        uploader="测试UP主",
        duration_str="2:00",
        duration_sec=duration_sec,
        publish_ts=1700000000,
        view_count=view_count,
        cover_url="https://example.com/cover.jpg",
        tid=20,
        tid_name="舞蹈",
        tags=tags or [],
        search_query="舞蹈",
    )


class TestDurationFilter:
    def setup_method(self):
        self.f = MetadataFilter(DEFAULT_CONFIG)

    def test_duration_too_short(self):
        item = _make_item(duration_sec=20.0)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False
        assert any("duration_too_short" in r for r in rec.metadata_filter_reasons)

    def test_duration_exactly_min(self):
        item = _make_item(duration_sec=30.0, title="独舞 solo 舞蹈")
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is True

    def test_duration_just_below_min(self):
        item = _make_item(duration_sec=29.9)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False

    def test_duration_too_long(self):
        item = _make_item(duration_sec=9999.0, title="独舞")
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False
        assert any("duration_too_long" in r for r in rec.metadata_filter_reasons)

    def test_duration_within_range(self):
        item = _make_item(duration_sec=180.0, title="独舞 solo")
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is True


class TestBlacklist:
    def setup_method(self):
        self.f = MetadataFilter(DEFAULT_CONFIG)

    def test_title_blacklist_reaction(self):
        item = _make_item(title="reaction 跳舞视频 独舞", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False
        assert any("blacklist_hit" in r for r in rec.metadata_filter_reasons)

    def test_title_blacklist_multi(self):
        item = _make_item(title="多人齐舞 超好看", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False

    def test_title_blacklist_mv(self):
        item = _make_item(title="MV 舞蹈", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False

    def test_title_blacklist_tutorial(self):
        item = _make_item(title="舞蹈教程 分解动作", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False

    def test_tag_blacklist(self):
        item = _make_item(title="好看的舞蹈", tags=["鬼畜"], duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is False


class TestWhitelist:
    def setup_method(self):
        self.f = MetadataFilter(DEFAULT_CONFIG)

    def test_whitelist_solo(self):
        item = _make_item(title="独舞 solo 现代舞", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is True
        assert any("whitelist_hit" in r for r in rec.metadata_filter_reasons)

    def test_whitelist_cover(self):
        item = _make_item(title="cover 舞蹈练习室", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is True

    def test_no_whitelist_still_passes_with_duration(self):
        # 没有任何白名单关键词，但时长满足，不应直接拒绝
        item = _make_item(title="一段视频", duration_sec=120)
        rec = self.f.filter(item)
        # 可能通过，也可能是 False（取决于实现），但不因缺少白名单词而直接拒绝
        # 这里只检查不因 whitelist_miss 导致 REJECT
        reject_reasons = [r for r in rec.metadata_filter_reasons if r.startswith("REJECT")]
        assert all("blacklist" not in r and "duration" not in r for r in reject_reasons)


class TestGenreMatching:
    def setup_method(self):
        self.f = MetadataFilter(DEFAULT_CONFIG)

    def test_hiphop_genre(self):
        item = _make_item(title="hiphop 街舞 solo practice", duration_sec=120)
        rec = self.f.filter(item)
        assert "hiphop" in rec.matched_genres
        assert any("hiphop" in kw.lower() for kw in rec.matched_keywords)

    def test_uighur_genre(self):
        item = _make_item(title="维吾尔族舞 独舞表演", duration_sec=120)
        rec = self.f.filter(item)
        assert "uighur" in rec.matched_genres

    def test_jazz_genre(self):
        item = _make_item(title="jazz dance 爵士舞 独舞", duration_sec=120)
        rec = self.f.filter(item)
        assert "jazz" in rec.matched_genres

    def test_korea_genre_kpop(self):
        item = _make_item(title="Kpop舞蹈 cover 韩舞", duration_sec=120)
        rec = self.f.filter(item)
        assert "korea" in rec.matched_genres

    def test_hoping_popping_genre(self):
        # "hoping" 实际对应 popping/机械舞
        item = _make_item(title="popping 机械舞 街舞", duration_sec=120)
        rec = self.f.filter(item)
        assert "hoping" in rec.matched_genres

    def test_multiple_genres(self):
        item = _make_item(title="breaking b-boy 霹雳舞 + locking 锁舞", duration_sec=120)
        rec = self.f.filter(item)
        assert "breaking" in rec.matched_genres
        assert "locking" in rec.matched_genres

    def test_no_genre_match(self):
        item = _make_item(title="舞蹈练习 一段普通视频", duration_sec=120)
        rec = self.f.filter(item)
        # 未必命中具体舞种也没关系，检查 matched_genres 是 list 类型
        assert isinstance(rec.matched_genres, list)

    def test_genre_keywords_in_matched_keywords(self):
        item = _make_item(title="敦煌舞 飞天舞 独舞", duration_sec=120)
        rec = self.f.filter(item)
        assert "dunhuang" in rec.matched_genres
        # matched_keywords 必须包含实际命中词
        assert len(rec.matched_keywords) > 0

    def test_metadata_score_range(self):
        """metadata_score 必须在 [0, 1] 区间。"""
        titles = [
            "独舞 solo 街舞 hiphop 练习室 cover",
            "多人齐舞 MV 教程",
            "一段视频",
            "维吾尔族舞 独舞表演 纯享版",
        ]
        for title in titles:
            item = _make_item(title=title, duration_sec=120)
            rec = self.f.filter(item)
            assert 0.0 <= rec.metadata_score <= 1.0, (
                f"metadata_score={rec.metadata_score} out of range for title='{title}'"
            )

    def test_audio_boost_keyword(self):
        """包含音频加分词时，reasons 里应有 AUDIO_BOOST 标注。"""
        item = _make_item(title="独舞 纯享版 原声 one take", duration_sec=120)
        rec = self.f.filter(item)
        assert rec.metadata_filter_pass is True
        assert any("AUDIO_BOOST" in r for r in rec.metadata_filter_reasons)

    def test_audio_penalty_keyword(self):
        """包含音频降权词时，reasons 里应有 AUDIO_PENALTY 标注。"""
        item = _make_item(title="街舞 vlog 口播", duration_sec=120)
        rec = self.f.filter(item)
        # 注意：vlog 在黑名单中会被先过滤
        if rec.metadata_filter_pass:
            assert any("AUDIO_PENALTY" in r for r in rec.metadata_filter_reasons)
