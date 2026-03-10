"""
test_dedup.py
=============
单元测试：去重逻辑（bvid 精确去重 + 标题相似度去重）
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bili_solo_pipeline.dedup import Deduplicator, _jaccard_similarity, _tokenize
from bili_solo_pipeline.models import VideoRecord
from bili_solo_pipeline.config import PipelineConfig, DEFAULT_CONFIG


def _make_record(
    bvid: str,
    title: str = "测试视频",
    cover_url: str = "https://example.com/cover.jpg",
) -> VideoRecord:
    return VideoRecord(
        bvid=bvid,
        title=title,
        url=f"https://www.bilibili.com/video/{bvid}",
        uploader="测试UP",
        duration_sec=120.0,
        metadata_filter_pass=True,
        cover_url=cover_url,
    )


class TestBvidDedup:
    def setup_method(self):
        self.dedup = Deduplicator(DEFAULT_CONFIG)

    def test_no_duplicates(self):
        records = [
            _make_record("BV001", "独舞视频一"),
            _make_record("BV002", "独舞视频二"),
            _make_record("BV003", "独舞视频三"),
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 3
        assert result.removed_count == 0
        assert len(result.duplicate_map) == 0

    def test_exact_bvid_dup(self):
        records = [
            _make_record("BV001", "标题一"),
            _make_record("BV002", "标题二"),
            _make_record("BV001", "标题一（重复）"),  # 重复
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 2
        assert result.removed_count == 1
        assert "BV001" in result.duplicate_map   # 第二个 BV001 -> 第一个

    def test_multiple_bvid_dups(self):
        records = [
            _make_record("BV001", "独舞视频A"),
            _make_record("BV001", "独舞视频A"),  # bvid 重复
            _make_record("BV001", "独舞视频A"),  # bvid 重复
            _make_record("BV002", "独舞视频B"),  # 不同 bvid + 不同标题
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 2
        assert result.removed_count == 2

    def test_empty_input(self):
        result = self.dedup.dedup([])
        assert len(result.unique_records) == 0
        assert result.removed_count == 0

    def test_single_record(self):
        records = [_make_record("BV001", "唯一记录")]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 1


class TestTitleDedup:
    def setup_method(self):
        # 设置较低的相似度阈值以方便测试
        cfg = PipelineConfig(title_sim_threshold=0.85, enable_phash_dedup=False)
        self.dedup = Deduplicator(cfg)

    def test_identical_titles_different_bvid(self):
        """完全相同的标题（不同 bvid）应被去重。"""
        records = [
            _make_record("BV001", "独舞视频 街舞 练习室版"),
            _make_record("BV002", "独舞视频 街舞 练习室版"),  # 完全相同
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 1
        assert result.removed_count == 1

    def test_very_similar_titles(self):
        """高度相似标题应被去重。"""
        records = [
            _make_record("BV001", "独舞视频 街舞 练习室版 2024"),
            _make_record("BV002", "独舞视频 街舞 练习室版 2024"),  # 完全相同
        ]
        result = self.dedup.dedup(records)
        assert result.removed_count >= 1

    def test_different_titles_kept(self):
        """明显不同的标题不应被去重。"""
        records = [
            _make_record("BV001", "维吾尔族舞 独舞表演"),
            _make_record("BV002", "hiphop 街舞 cover"),
            _make_record("BV003", "爵士舞 jazz 练习室"),
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 3

    def test_bvid_dedup_takes_priority(self):
        """bvid 去重应在标题去重之前执行（相同 bvid 即使标题不同也去重）。"""
        records = [
            _make_record("BV001", "标题版本一"),
            _make_record("BV001", "标题版本二（完全不同）"),
        ]
        result = self.dedup.dedup(records)
        assert len(result.unique_records) == 1


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert _jaccard_similarity("独舞视频", "独舞视频") == pytest.approx(1.0)

    def test_completely_different(self):
        sim = _jaccard_similarity("独舞视频街舞", "综艺节目采访")
        assert sim < 0.3

    def test_empty_strings(self):
        assert _jaccard_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert _jaccard_similarity("独舞视频", "") == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = _jaccard_similarity("独舞视频 街舞版", "独舞视频 练习室版")
        assert 0.3 < sim < 0.9  # 部分重叠

    def test_english_tokens(self):
        sim = _jaccard_similarity("hiphop dance cover", "hiphop dance practice")
        # 交集={hiphop,dance}=2，并集={hiphop,dance,cover,practice}=4 → jaccard=0.5
        assert sim >= 0.5  # 有两个共同词


class TestTokenize:
    def test_chinese_bigrams(self):
        tokens = _tokenize("独舞")
        assert "独舞" in tokens

    def test_english_words(self):
        tokens = _tokenize("hiphop dance")
        assert "hiphop" in tokens
        assert "dance" in tokens

    def test_mixed(self):
        tokens = _tokenize("hiphop街舞 cover")
        assert "hiphop" in tokens
        assert "cover" in tokens
        assert "街舞" in tokens


class TestDedupPreserveOrder:
    """验证去重保留第一个出现的记录（按输入顺序优先保留）。"""

    def test_first_occurrence_preserved(self):
        dedup = Deduplicator(DEFAULT_CONFIG)
        records = [
            _make_record("BV001", "第一个出现"),
            _make_record("BV001", "第二个出现（重复）"),
        ]
        result = dedup.dedup(records)
        assert len(result.unique_records) == 1
        assert result.unique_records[0].title == "第一个出现"
