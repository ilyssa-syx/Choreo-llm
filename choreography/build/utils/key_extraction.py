from essentia.standard import MonoLoader, KeyExtractor
"""
Key Analysis
return: str
"""
def extract_key(audio_path, start_sec, end_sec, sample_rate=44100):
    loader = MonoLoader(filename=audio_path, sampleRate=sample_rate)
    audio = loader()  # 返回 NumPy array，shape=(n_samples,)
    print(audio.shape)
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # 截取指定时间段
    audio_segment = audio[start_sample:end_sample]

    extractor = KeyExtractor()
    key, scale, strength = extractor(audio_segment)
    key_label = f"{key} {scale}"
    return key_label

