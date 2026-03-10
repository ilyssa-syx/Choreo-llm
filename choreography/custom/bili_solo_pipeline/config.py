"""
config.py
=========
全局配置：阈值、舞种词表、黑白名单。

所有参数均可在 CLI 中通过 --config_override JSON 字符串覆盖，
也可通过代码直接修改 DEFAULT_CONFIG 的对应字段（适合研究场景快速调参）。

舞种命名说明：
  - "hoping"  实际是 popping/poppin（机械舞/震感舞），词表名为内部训练分类名，代码注释说明映射
  - "tai"     含傣族舞和泰国舞，中英文均有歧义，采用多关键词兜底
  - "korea"   含 K-POP 偶像舞和朝鲜族民间舞，两类差异大，均保留供人工确认
  - "shenyun"  指中国古典舞"身韵"技法，非神韵艺术团，关键词需注意区分
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 舞种 -> 搜索关键词映射
# 每组关键词在一次查询中 OR，不同舞种分批次搜索以提高召回
# ---------------------------------------------------------------------------

GENRE_KEYWORD_MAP: dict[str, list[str]] = {
    # === 街舞系 ===
    "urban": [
        "urban", "urban dance", "urban编舞", "urban舞蹈", "都市编舞",
    ],
    # 注意：内部标签名为 "hoping"，实为 popping/poppin（机械舞/震感舞）
    # Bilibili 上常见写法：popping、poppin、机械舞、震感舞、电流舞
    "hoping": [
        "hoping", "popping", "poppin",
        "机械舞", "震感舞", "电流舞",
    ],
    "hiphop": [
        "hiphop", "hip-hop", "hip hop",
        "嘻哈舞", "街舞hiphop", "hiphop编舞",
    ],
    "breaking": [
        "breaking", "breakdance", "b-boy", "b-girl",
        "霹雳舞", "地板动作",
    ],
    "locking": [
        "locking", "lockin", "lock",
        "锁舞", "街舞locking",
    ],
    "jazz": [
        "jazz", "jazz dance",
        "爵士舞", "jazz编舞", "街舞爵士",
    ],

    # === 西方音乐风格舞蹈 ===
    # "classical": 西方古典舞/芭蕾，与中国古典舞（hantang/shenyun）区分
    "classical": [
        "ballet", "ballet dance", "classical ballet", "variation ballet",
        "芭蕾", "芭蕾舞", "古典芭蕾", "芭蕾独舞", "芭蕾 solo",
        "芭蕾编舞", "芭蕾舞者", "足尖", "足尖舞", "足尖练习", "pointe",
        "剧院芭蕾", "舞剧芭蕾", "芭蕾片段", "芭蕾课堂", "芭蕾排练",
        "芭蕾变奏", "芭蕾舞剧", "天鹅湖", "胡桃夹子", "睡美人", "吉赛尔",
    ],
    # "country": 乡村舞/摇摆舞风格
    "country": [
        "country dance", "乡村舞", "牛仔舞", "乡村风格舞蹈",
        "country swing", "country舞", "乡村摇摆舞",
        "line dance", "country line dance", "排舞", "乡村排舞", "线舞", "cowboy dance",
    ],
    # "metal": 金属乐风格编舞，在B站常与街舞/电音舞蹈混合出现
    "metal": [
        "metal dance", "金属乐舞蹈", "metal风格编舞",
        "heavy metal舞蹈", "metal舞", "摇滚金属舞蹈",
    ],
    # "reggae": 雷鬼舞/牙买加风格舞蹈
    "reggae": [
        "reggae", "reggae dance", "雷鬼", "雷鬼舞",
        "雷鬼舞蹈", "reggae舞蹈", "雷鬼风格",
    ],
    # "rock": 摇滚风格舞蹈编舞（rockandroll、摇滚舞）
    "rock": [
        "rock dance", "摇滚舞", "rock舞蹈", "rockandroll舞蹈",
        "rock编舞", "摇滚风格编舞", "rock and roll舞",
    ],
    # "disco": 迪斯科舞，70-80年代复古风格
    "disco": [
        "disco", "disco dance", "迪斯科", "迪斯科舞",
        "迪斯科舞蹈", "disco编舞", "复古迪斯科", "迪斯科风格",
    ],
    # "blues": 布鲁斯/蓝调舞蹈
    "blues": [
        "blues dance", "布鲁斯舞", "蓝调舞蹈", "blues舞",
        "布鲁斯", "蓝调舞", "blues编舞",
    ],

    # === 民族舞系 ===
    # "uighur": 维吾尔族舞，注意"新疆舞"用词较宽泛，包含汉族表演风格
    "uighur": [
        "维吾尔族舞", "维族舞", "新疆舞", "维吾尔舞蹈",
        "维吾尔舞", "新疆民族舞", "维吾尔族舞蹈",
    ],
    # "tai": 傣族舞 + 泰国舞，两者风格相似，命名有歧义
    "tai": [
        "傣族舞", "傣舞", "孔雀舞",
        "民族舞 傣族", "泰国舞", "泰式舞蹈",
        "傣族舞蹈", "傣族孔雀舞", "泰国舞蹈", "泰国传统舞",
    ],
    # "korea": 包含 K-POP 偶像舞（现代）和朝鲜族民间舞（传统），差异显著
    # 保留两类以提高召回，label 阶段可由人工进一步细分
    "korea": [
        "韩舞", "Kpop舞蹈", "K-POP cover",
        "韩舞 cover", "韩国舞蹈",
        "朝鲜族舞", "民族舞 朝鲜族", "朝鲜族舞蹈",
    ],
    "hmong": [
        "苗族舞", "苗族舞蹈", "苗舞", "民族舞 苗族",
        "苗族芦笙舞", "芦笙舞",
    ],

    # === 古典舞系 ===
    "hantang": [
        "汉唐舞", "汉唐古典舞", "汉唐风舞蹈", "汉唐舞蹈",
        "汉唐古典", "汉唐风", "汉唐",
    ],
    # "shenyun": 指中国古典舞"身韵"（舞蹈技法），与神韵艺术团(Shen Yun)不同
    # 避免搜索到"神韵"艺术团相关内容，关键词以"身韵"为主
    "shenyun": [
        "身韵", "古典舞身韵", "中国古典舞身韵", "身韵组合", "身韵练习",
        "身韵训练", "身韵组合训练",
    ],
    "dunhuang": [
        "敦煌舞", "敦煌飞天舞", "敦煌古典舞", "敦煌舞蹈", "飞天舞",
        "敦煌飞天", "敦煌壁画舞", "壁画舞",
    ],
}

# 通用舞蹈倾向词（单独搜索或与舞种词组合）
GENERAL_DANCE_KEYWORDS: list[str] = [
    "舞蹈", "编舞", "独舞", "solo", "个人", "个人cut",
    "纯享", "全身", "练习室", "cover", "one take", "一镜到底",
]

# ---------------------------------------------------------------------------
# 标题/标签 黑白名单
# ---------------------------------------------------------------------------

TITLE_WHITELIST: list[str] = [
    "舞蹈", "独舞", "solo", "个人", "纯享", "全身",
    "练习室", "cover", "one take", "一镜到底",
    "现代舞", "街舞", "中国舞", "民族舞", "古典舞",    # 比赛/选秀摄像水平更高，应重点收录
    "比赛", "选秀",    # 舞种关键词自动合并（见 metadata_filter.py）
]

# 硬性黑名单：命中即直接拒绝（明确不含独舞内容）
TITLE_BLACKLIST: list[str] = [
    "多人", "齐舞", "团舞", "reaction", "解说", "盘点",
    "采访", "花絮", "预告", "混剪", "剪辑", "鬼畜",
    "MAD", "口播", "vlog", "综艺", "节目片段",
]

# 软性降权词：出现后降低 metadata_score，但不硬性拒绝
# 这些内容「可能」含有单人独舞（如 MV 独舞镜头、晚会个人节目）
TITLE_SOFT_PENALTY: list[str] = [
    "MV", "教程", "教学", "拆解", "晚会", "春晚",
]

# 音频相关加分词（出现在标题/标签中）
AUDIO_BOOST_KEYWORDS: list[str] = [
    "原声", "高音质", "无解说", "纯享", "练习室",
    "无字幕", "live", "one take", "无剪辑",
]

# 音频相关降权词
AUDIO_PENALTY_KEYWORDS: list[str] = [
    "口播", "采访", "讲解", "vlog", "收音差", "杂音",
    "现场太吵", "嘈杂", "麦克风故障", "音效差",
]

# ---------------------------------------------------------------------------
# 默认数值阈值（可通过 CLI 或 config_override 覆盖）
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # ---- 搜索参数 ----
    max_pages_per_query: int = 5          # 每个关键词最多搜索页数
    results_per_page: int = 20            # 每页结果数（B 站通常 20）
    search_order: str = "totalrank"       # totalrank / click / pubdate / dm / stow
    request_delay_sec: float = 1.5        # 每次请求最小间隔（秒）
    max_retries: int = 3                  # 最大重试次数
    target_total_hours: float = 0.0       # 搜索目标总时长（小时）；0 = 不限制
    search_oversample_rate: float = 4.0   # 搜原始素材是目标的 N 倍（弥补过滤损耗）
    max_pages_when_targeting: int = 50    # target_total_hours > 0 时自动扩展的单关键词页数上限
    retry_base_delay_sec: float = 2.0     # 指数退避基础延迟（秒）
    search_timeout_sec: int = 15          # 单次请求超时（秒）

    # ---- 元数据过滤 ----
    min_duration_sec: float = 30.0        # 最短时长
    max_duration_sec: float = 3600.0      # 最长时长（过长可能是合集）
    min_view_count: int = 0               # 最低播放量（0 = 不限）

    # ---- 视觉检测 ----
    sample_every_sec: float = 1.0         # 抽帧间隔（秒）
    max_frames: int = 60                  # 最大抽帧数
    skip_head_sec: float = 3.0            # 跳过片头（秒）
    yolo_model: str = "yolov8n.pt"        # YOLO 模型权重（n/s/m/l/x）
    yolo_conf_threshold: float = 0.4      # 检测置信度阈值
    enable_pose: bool = False             # 是否启用姿态估计

    # ---- Sol 评分 ----
    solo_threshold: float = 0.75          # >=此值 -> keep（视觉层面）
    uncertain_threshold: float = 0.45    # >=此值 -> review；<此值 -> drop
    min_valid_person_ratio: float = 0.3  # 有效人体帧最低占比（否则为 no_person）
    min_single_person_ratio: float = 0.5 # 单人帧最低占比（solo 判断核心）
    max_crowded_ratio: float = 0.3        # 多人帧最大允许占比
    min_dominant_area_ratio: float = 0.04 # 主体人物面积最小占比（画面 4%）

    # ---- 音频评分 ----
    enable_audio_scoring: bool = True
    audio_threshold: float = 0.60         # >=此值 -> audio 通过
    speech_ratio_max: float = 0.35        # 语音占比超过此值 -> speech_heavy
    min_audio_present_ratio: float = 0.3  # 非静音最低占比（避免空音轨）
    min_loudness_mean: float = -45.0      # 最低平均响度 LUFS（过低 -> low_volume）
    max_clipping_ratio: float = 0.02      # 最大削波比例（超过 -> noisy）
    use_heuristic_audio: bool = True      # 无模型时使用启发式音频评估
    audio_segment_duration_sec: float = 30.0  # 取样音频段长度（秒，从中间取）

    # ---- 去重 ----
    enable_phash_dedup: bool = False      # 封面图感知哈希去重（需额外依赖）
    title_sim_threshold: float = 0.85    # 标题归一化相似度阈值

    # ---- 融合排序 ----
    weight_solo: float = 0.55
    weight_audio: float = 0.30
    weight_metadata: float = 0.15
    # 若视觉不稳定但音频清晰（如多机位节目），进入 review 而非 drop
    unstable_vision_audio_rescue: bool = True

    # ---- 下载 ----
    enable_download: bool = False          # 是否在 pipeline 末自动下载视频
    download_dir: str = "data/videos"      # 视频下载根目录
    download_min_fusion_score: float = 0.0 # 仅下载 final_score 达到此值的视频（0=不限）
    download_keep: bool = True             # 下载 label=keep 的视频
    download_review: bool = False          # 下载 label=review 的视频
    download_workers: int = 3              # 并发下载线程数
    download_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    download_limit_rate: str = ""          # 限速（空=不限，如 '5M' 限 5MB/s）
    download_cookies_file: str = ""        # yt-dlp 使用的 cookies 文件

    # ---- 断点续跑 ----
    use_cache: bool = True
    cache_dir: str = ".pipeline_cache"

    # ---- 输出 ----
    out_dir: str = "data/bili_solo_candidates"
    write_keep: bool = True
    write_review: bool = True
    write_drop: bool = True
    write_summary_csv: bool = True

    # ---- 杂项 ----
    log_level: str = "INFO"
    queries_file: Optional[str] = None    # 自定义关键词文件路径
    use_genre_keywords: bool = True       # 是否使用内置舞种词表
    use_general_keywords: bool = True     # 是否追加通用舞蹈词


DEFAULT_CONFIG = PipelineConfig()
