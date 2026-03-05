import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from pathlib import Path

def process_features(feat, n_clusters=5):
    
    
    sim_matrix = 1 - pairwise_distances(feat, metric='cosine')
            
    T = feat.shape[0]
            
    idx = np.arange(T) / (T - 1)
    context_feat = np.concatenate((sim_matrix, idx.reshape(T, 1)), axis=1)
            
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_s = kmeans.fit_predict(context_feat)
            
    return labels_s
