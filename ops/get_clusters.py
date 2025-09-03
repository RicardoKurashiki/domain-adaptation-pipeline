import os
import numpy as np
from utils import load_data, split_data_by_label
from sklearn.cluster import KMeans

def get_clusters(X, y, K=1, cluster_dir="./checkpoint/data/clusters/", seed=42):
    result = {}
    data_by_labels = split_data_by_label(X, y)
    
    for label in data_by_labels:
        save_path = os.path.join(cluster_dir, f"{label}/")
        os.makedirs(save_path, exist_ok=True)
        
        if len(data_by_labels[label]) < K:
            print(f"Warning: Not enough data for class {label} to create {K} clusters. Using KMeans with fewer clusters.")
            actual_k = max(1, len(data_by_labels[label]))
            if actual_k == 0:
                print(f"Warning: Class {label} has no data. Skipping clustering.")
                continue
            kmeans = KMeans(n_clusters=actual_k, random_state=seed, n_init='auto').fit(data_by_labels[label])
        else:
            kmeans = KMeans(n_clusters=K, random_state=seed, n_init='auto').fit(data_by_labels[label])
        
        for _, centroid in enumerate(kmeans.cluster_centers_):
            np.save(os.path.join(save_path, f"centroid.npy"), centroid)
        
        result[label] = kmeans.cluster_centers_
    return result

def load_clusters(checkpoint_dir="./checkpoint/", split="train"):
    clusters_dir = os.path.join(checkpoint_dir, "clusters", f"{split}/")
    if not os.path.exists(clusters_dir):
        return None
    result = {}
    for file in os.listdir(clusters_dir):
        if file.startswith("."):
            continue
        if os.path.isdir(os.path.join(clusters_dir, file)):
            result[file] = np.load(os.path.join(clusters_dir, file, "centroid.npy"))
    return result


def main(data_dir, K=1, checkpoint_dir="./checkpoint/", seed=42, splits=["train", "val", "test"]):
    for split in splits:
        X, y = load_data(data_dir, split=split)
        clusters_dir = os.path.join(checkpoint_dir, "clusters", f"{split}/")
        os.makedirs(clusters_dir, exist_ok=True)
        get_clusters(X, y, K, clusters_dir, seed)