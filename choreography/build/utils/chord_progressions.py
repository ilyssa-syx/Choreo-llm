from madmom.features.chords import CNNChordFeatureProcessor  
from madmom.features.chords import CRFChordRecognitionProcessor

"""
Chord Progression Analysis using madmom
return: str
"""
def extract_chord_progression(audio_path, start_time=0.0, end_time=0.0):
    # 第一步:提取深度 chroma 特征
    featproc = CNNChordFeatureProcessor()
    decode = CRFChordRecognitionProcessor()
    filename1= audio_path
    feats = featproc(filename1)
    result = decode(feats)
    # print(result)
    filtered_chords = []
    for (start, end, chord) in result:
        if start >= start_time and end <= end_time:
            filtered_chords.append(chord)
    
    # 返回用 '-' 连接的和弦字符串
    return '-'.join(filtered_chords)

# extract_chord_progression("wavs/mBR0.mp3")