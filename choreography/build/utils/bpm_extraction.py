# id -> tempo (BPM)
id2tempo = {
    # Break
    "mBR0": 80,  "mBR1": 90,  "mBR2": 100, "mBR3": 110, "mBR4": 120, "mBR5": 130,
    # Pop
    "mPO0": 80,  "mPO1": 90,  "mPO2": 100, "mPO3": 110, "mPO4": 120, "mPO5": 130,
    # Lock
    "mLO0": 80,  "mLO1": 90,  "mLO2": 100, "mLO3": 110, "mLO4": 120, "mLO5": 130,
    # Middle Hip-hop
    "mMH0": 80,  "mMH1": 90,  "mMH2": 100, "mMH3": 110, "mMH4": 120, "mMH5": 130,
    # LA style Hip-hop
    "mLH0": 80,  "mLH1": 90,  "mLH2": 100, "mLH3": 110, "mLH4": 120, "mLH5": 130,
    # House
    "mHO0": 110, "mHO1": 115, "mHO2": 120, "mHO3": 125, "mHO4": 130, "mHO5": 135,
    # Waack
    "mWA0": 80,  "mWA1": 90,  "mWA2": 100, "mWA3": 110, "mWA4": 120, "mWA5": 130,
    # Krump
    "mKR0": 80,  "mKR1": 90,  "mKR2": 100, "mKR3": 110, "mKR4": 120, "mKR5": 130,
    # Street Jazz
    "mJS0": 80,  "mJS1": 90,  "mJS2": 100, "mJS3": 110, "mJS4": 120, "mJS5": 130,
    # Ballet Jazz
    "mJB0": 80,  "mJB1": 90,  "mJB2": 100, "mJB3": 110, "mJB4": 120, "mJB5": 130,
}

def get_tempo(track_id: str, default=None):
    """返回 track_id 对应的 BPM（整数），找不到返回 default（默认为 None）。"""
    return id2tempo.get(track_id, default)
