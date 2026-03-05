import librosa
import numpy as np
from pathlib import Path
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


song_name_list = [
    "gBR_sBM_cAll_d04_mBR1_ch03_slice0",
    "gHO_sBM_cAll_d19_mHO0_ch03_slice0",
    "gJB_sBM_cAll_d07_mJB0_ch03_slice0",
    "gJS_sBM_cAll_d01_mJS0_ch03_slice0",
    "gKR_sBM_cAll_d28_mKR0_ch03_slice0",
    "gLH_sBM_cAll_d16_mLH0_ch03_slice0",
    "gLO_sBM_cAll_d13_mLO0_ch09_slice8",
    "gMH_sBM_cAll_d22_mMH0_ch10_slice7",
    "gPO_sBM_cAll_d10_mPO0_ch10_slice10",
    "gWA_sBM_cAll_d26_mWA1_ch04_slice4"
]


song_name_list_new = [
    "gBR_sBM_cAll_d04_mBR0_ch02_slice6",
    "gHO_sBM_cAll_d21_mHO5_ch02_slice2",
    "gJB_sBM_cAll_d09_mJB5_ch02_slice1",
    "gJS_sBM_cAll_d03_mJS3_ch02_slice0",
    "gKR_sBM_cAll_d28_mKR2_ch02_slice9",
    "gLH_sBM_cAll_d18_mLH4_ch02_slice2",
    "gLO_sBM_cAll_d13_mLO2_ch02_slice5",
    "gMH_sBM_cAll_d24_mMH3_ch02_slice4",
    "gPO_sBM_cAll_d11_mPO1_ch02_slice5",
    "gWA_sBM_cAll_d26_mWA0_ch02_slice8"
]

# 加载音频文件
def process_audio(audio_path, output_path):
    y, sr = librosa.load(audio_path)
    y = y[:int(len(y) * 24/25)]

    print(f"采样率: {sr} Hz")
    print(f"音频时长: {len(y)/sr:.2f} 秒")

    # 提取节拍和速度
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    print(f"\n估计速度 (BPM): {tempo:.2f}")
    print(f"检测到的节拍数: {len(beat_frames)}")
    print("节拍：", beat_frames)

    # 将帧转换为时间（秒）
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    audio_duration = len(y) / sr
    beat_frame_indexes = (beat_times / audio_duration * 144).astype(int)
    np.save(output_path, beat_frame_indexes)


def process_audio_madmom(audio_path, output_path, n_frames=144, fps=100):
    """
    用 madmom 检测节拍并把节拍时间转换为 [0, n_frames-1] 的索引数组，保存为 .npy
    
    参数:
      audio_path: 输入音频路径
      output_path: 输出 .npy 路径
      n_frames: 目标帧数（你示例中是 144）
      fps: madmom DBN 的 fps（通常默认 100 就可）
    """
    # 读取音频以获取采样率并模拟你之前的裁剪（24/25）
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    trimmed_samples = int(len(y) * 24 / 25)
    trimmed_dur = trimmed_samples / sr

    # madmom 检测：RNN -> DBN
    act = RNNBeatProcessor()(audio_path)            # 读取原文件并产生 activation
    proc = DBNBeatTrackingProcessor(fps=fps)        # fps 可调整
    beats_sec = proc(act)                           # 返回以秒为单位的节拍时间数组

    # 只保留发生在裁剪后时长内的节拍（和 librosa 行为一致）
    beats_sec = np.asarray(beats_sec)
    beats_sec = beats_sec[beats_sec <= trimmed_dur + 1e-8]

    # 映射到 0..n_frames-1（和你用 librosa 的公式一致）
    # 注意：当 beat == trimmed_dur 时，可能会映射到 n_frames -> 我们用 clip 限制到 n_frames-1
    raw_indexes = (beats_sec / trimmed_dur * n_frames).astype(int)
    beat_frame_indexes = np.clip(raw_indexes, 0, n_frames - 1).astype(int)

    np.save(output_path, beat_frame_indexes)
    print(f"[madmom] 检测到 {len(beat_frame_indexes)} 个节拍，保存到 {output_path}")

def detect_downbeats_madmom(audio_path, output_path, n_frames=144, trimmed_ratio=24/25, fps=100):
    """
    使用 madmom 检测 downbeat（小节起点），并映射到 0..n_frames-1 的索引（与你现有 pipeline 保持一致）。
    返回：
        downbeat_times: 小节起点的秒数数组
        downbeat_frame_indexes: 映射到 0..n_frames-1 的整数索引数组（clip 到合法范围）
    说明：
        madmom 的具体返回形式和版本可能有差异，因此这里用了 try/except 做兼容处理。
    """
    # 读取以获取裁剪后时长（和你之前使用的 24/25 一致）
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    trimmed_samples = int(len(y) * trimmed_ratio)
    trimmed_dur = trimmed_samples / sr if sr and len(y)>0 else 0.0

    # madmom downbeat pipeline
    act = RNNDownBeatProcessor()(audio_path)   # activation
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=fps)
    # proc(act) 在常见版本中返回每个 downbeat 的时间（秒），有些版本返回 Nx2 (time, bar_pos)
    db_out = proc(act)
    db_out = np.asarray(db_out)

    # 只取 bar_pos == 1 的行
    db_times = db_out[db_out[:, 1] == 1][:, 0]

    # 只保留在 trimmed_dur 以内的 downbeats
    if db_times.size > 0 and trimmed_dur > 0:
        db_times = db_times[db_times <= trimmed_dur + 1e-8]

    # 映射到 0..n_frames-1
    if trimmed_dur <= 0:
        frame_indexes = np.array([], dtype=int)
    else:
        raw_idx = (db_times / trimmed_dur * n_frames).astype(int)
        frame_indexes = np.clip(raw_idx, 0, n_frames-1).astype(int)

    np.save(output_path, frame_indexes)
    print(f"[madmom] 检测到 {len(frame_indexes)} 个节拍，保存到 {output_path}")

def detect_downbeats_beatnet(audio_path, output_path, n_frames=144, trimmed_ratio=24/25, model=3):
    """
    使用 BeatNet 检测 downbeat（小节起点），并映射到 0..n_frames-1 的索引。
    
    参数：
        audio_path: 音频文件路径
        output_path: 输出 .npy 文件路径
        n_frames: 目标帧数（默认 144）
        trimmed_ratio: 音频裁剪比例（默认 24/25）
        model: BeatNet 模型选择（1, 2, 或 3，默认 1）
    
    返回：
        downbeat_times: 小节起点的秒数数组
        downbeat_frame_indexes: 映射到 0..n_frames-1 的整数索引数组
    
    说明：
        BeatNet 返回 (beats, downbeats) 元组，downbeats 是小节起点的时间戳数组。
    """
    
    # 读取音频以获取裁剪后时长
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    trimmed_samples = int(len(y) * trimmed_ratio)
    trimmed_dur = trimmed_samples / sr if sr and len(y) > 0 else 0.0
    
    # 初始化 BeatNet
    # model: 1=online, 2=offline, 3=offline strategic
    estimator = BeatNet(model=model, mode='offline', inference_model='DBN', 
                       plot=[], thread=False)
    
    # 处理音频文件
    # BeatNet 返回 (beats, downbeats)，其中 downbeats 是小节起点时间数组
    output = estimator.process(audio_path)
    
    # 提取 downbeats
    if isinstance(output, tuple) and len(output) >= 2:
        db_times = np.asarray(output[1])  # downbeats 在第二个位置
    else:
        # 如果返回格式不同，尝试直接作为 downbeats
        db_times = np.asarray(output)
    
    # 确保是一维数组
    if db_times.ndim > 1:
        db_times = db_times.flatten()
    
    # 只保留在 trimmed_dur 以内的 downbeats
    if db_times.size > 0 and trimmed_dur > 0:
        db_times = db_times[db_times <= trimmed_dur + 1e-8]
    
    # 映射到 0..n_frames-1
    if trimmed_dur <= 0 or db_times.size == 0:
        frame_indexes = np.array([], dtype=int)
    else:
        raw_idx = (db_times / trimmed_dur * n_frames).astype(int)
        frame_indexes = np.clip(raw_idx, 0, n_frames - 1).astype(int)
    
    # 保存结果
    np.save(output_path, frame_indexes)
    print(f"[BeatNet] 检测到 {len(frame_indexes)} 个 downbeat，保存到 {output_path}")
    
    return db_times, frame_indexes

def detect_downbeats_from_full_audio(
    full_audio_path, 
    slice_index, 
    output_path, 
    n_frames=144, 
    slice_duration=5.0,
    trimmed_ratio=24/25, 
    model=3
):
    """
    从完整音频检测 downbeat，然后提取指定 slice 对应的 downbeats。
    
    参数：
        full_audio_path: 完整音频文件路径
        slice_index: slice 索引（例如 slice0 -> 0, slice8 -> 8）
        output_path: 输出 .npy 文件路径
        n_frames: 目标帧数（默认 144）
        slice_duration: 每个 slice 的时长（默认 5.0 秒）
        trimmed_ratio: slice 内音频裁剪比例（默认 24/25）
        model: BeatNet 模型选择（1, 2, 或 3，默认 3）
    
    返回：
        downbeat_times: slice 内小节起点的相对秒数数组
        downbeat_frame_indexes: 映射到 0..n_frames-1 的整数索引数组
    """
    
    # 计算 slice 的起始和结束时间
    slice_start_time = 0.5 * slice_index
    slice_end_time = slice_start_time + slice_duration
    
    # slice 内实际使用的时长（考虑 trimmed_ratio）
    trimmed_slice_duration = slice_duration * trimmed_ratio
    
    print(f"处理: {Path(full_audio_path).name}")
    print(f"  Slice {slice_index}: {slice_start_time:.2f}s - {slice_end_time:.2f}s")
    print(f"  裁剪后时长: {trimmed_slice_duration:.2f}s")
    
    # 初始化 BeatNet
    estimator = BeatNet(
        model=model, 
        mode='offline', 
        inference_model='DBN', 
        plot=[], 
        thread=False
    )
    
    # 处理完整音频文件
    print("  正在检测 downbeats...")
    output = estimator.process(full_audio_path)
    print(isinstance(output, tuple))
    print(output.shape)
    print(output)
    """
    downbeat_mask = output[:, 1] == 1
    all_db_times = output[downbeat_mask]
    """
    all_db_times = np.asarray(output)
    
    print(all_db_times.shape)
    
    all_db_times = all_db_times[:, 0]
    print(all_db_times)
    
    # 筛选出在当前 slice 时间范围内的 downbeats
    slice_db_times = all_db_times[
        (all_db_times >= slice_start_time) & 
        (all_db_times < slice_start_time + trimmed_slice_duration)
    ]
    
    print(f"  Slice {slice_index} 内有 {len(slice_db_times)} 个 downbeat")
    
    # 转换为相对于 slice 起始的时间
    relative_db_times = slice_db_times - slice_start_time
    
    # 映射到 0..n_frames-1
    if trimmed_slice_duration <= 0 or len(relative_db_times) == 0:
        frame_indexes = np.array([], dtype=int)
    else:
        raw_idx = (relative_db_times / trimmed_slice_duration * n_frames).astype(int)
        frame_indexes = np.clip(raw_idx, 0, n_frames - 1).astype(int)
    
    # 保存结果
    np.save(output_path, frame_indexes)
    print(f"  保存到: {output_path}")
    print(f"  Frame indexes: {frame_indexes}\n")
    
    
def process_song_list(
    song_name_list, 
    full_audio_dir, 
    output_dir,
    n_frames=144,
    slice_duration=5.0,
    trimmed_ratio=24/25,
    model=3
):
    """
    批量处理 song_name_list 中的所有音频切片
    
    参数:
        song_name_list: 包含完整 slice 名称的列表
        full_audio_dir: 存放完整音频文件的目录
        output_dir: 输出 .npy 文件目录
        n_frames: 目标帧数（默认 144）
        slice_duration: 每个 slice 的时长（默认 5.0 秒）
        trimmed_ratio: slice 内音频裁剪比例（默认 24/25）
        model: BeatNet 模型选择（默认 3）
    """
    full_audio_dir = Path(full_audio_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始处理 {len(song_name_list)} 个音频切片\n")
    print("=" * 60)
    
    for song_name in song_name_list:
        try:
            # 解析 song_name，提取基础名称和 slice 索引
            # 例如: "gBR_sBM_cAll_d04_mBR1_ch03_slice0"
            parts = song_name.split('_slice')
            if len(parts) != 2:
                print(f"跳过无效格式: {song_name}\n")
                continue
            
            base_name = parts[0]  # "gBR_sBM_cAll_d04_mBR1_ch03"
            slice_index = int(parts[1])  # 0, 8, 10, etc.
            
            # 构造完整音频文件路径
            full_audio_path = full_audio_dir / f"{base_name}.wav"
            
            if not full_audio_path.exists():
                print(f"找不到完整音频: {full_audio_path}\n")
                continue
            
            # 构造输出路径
            output_path = output_dir / f"{song_name}.npy"
            
            # 处理
            detect_downbeats_from_full_audio(
                full_audio_path=str(full_audio_path),
                slice_index=slice_index,
                output_path=str(output_path),
                n_frames=n_frames,
                slice_duration=slice_duration,
                trimmed_ratio=trimmed_ratio,
                model=model
            )
            
        except Exception as e:
            print(f"处理 {song_name} 时出错: {e}\n")
    
    print("=" * 60)
    print("处理完成！")


def process_all_audios(input_dir, output_dir):
    """
    处理目录下所有音频文件
    
    参数:
        input_dir: 输入音频文件目录
        output_dir: 输出.npy文件目录
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有音频文件
    audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    for audio_path in audio_files:
        try:
            # 保持相同的basename
            output_path = output_dir / f"{audio_path.stem}.npy"
            detect_downbeats_beatnet(str(audio_path), str(output_path))
            
        except Exception as e:
            print(f"处理 {audio_path} 时出错: {e}\n")

def main():
    # 处理test目录
    print('=== 处理 test 音频 ===')
    process_song_list(
        song_name_list=song_name_list_new,
        full_audio_dir='test_wavs_whole',  # 修改为你的完整音频目录
        output_dir='test_timedots_beatnet',
        n_frames=144,
        slice_duration=5.0,
        trimmed_ratio=24/25,
        model=3
    )
    
    # 处理train目录
    print('\n=== 处理 train 音频 ===')
    process_song_list(
        song_name_list=song_name_list,
        full_audio_dir='train_wavs_whole',  # 修改为你的完整音频目录
        output_dir='train_timedots_beatnet',
        n_frames=144,
        slice_duration=5.0,
        trimmed_ratio=24/25,
        model=3
    )
    
    print('全部完成！')

if __name__ == "__main__":
    main()




