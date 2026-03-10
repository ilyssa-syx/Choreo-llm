def extract_genre_from_filename(inp: str) -> str:
    GENRE_MAPPING = {
        'mBR': 'Break',
        'mPO': 'Pop',
        'mLO': 'Lock',
        'mWA': 'Waack',
        'mMH': 'Middle Hip-hop',
        'mLH': 'LA-style Hip-hop',
        'mHO': 'House',
        'mKR': 'Krump',
        'mJS': 'Street Jazz',
        'mJB': 'Ballet Jazz'
    }
    return GENRE_MAPPING.get(inp, "any")