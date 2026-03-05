# utils/dance_beat.py
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter as G

def extract_motion_beats(joints, starting_point):
    # Calculate velocity.
    seq_len = joints.shape[0]
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = argrelextrema(envelope, np.less, axis=0, order=5)[0] # 5 for 60FPS
    peak_idxs += starting_point
    return peak_idxs
