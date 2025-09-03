import os
from utils import load_data
from ops import get_pca

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_pca_plot(X, y, pca, split, domain, results_dir):
    X_pca = pca.transform(X)
    
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis")
    plt.title(f"PCA Plot - {domain} {split}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class", loc="best")
    plt.tight_layout()
    save_dir = os.path.join(results_dir, domain, f"{split}/")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "pca.png"))
    
    plt.close()

def get_joint_pca_plot(X_source, y_source, X_target, y_target, pca, split, results_dir):
    X_source_pca = pca.transform(X_source)
    X_target_pca = pca.transform(X_target)

    X_combined = np.vstack([X_source_pca, X_target_pca])
    domains = np.array(["source"] * X_source_pca.shape[0] + ["target"] * X_target_pca.shape[0])
    labels = np.concatenate([y_source, y_target])

    # Build combined labels (domain-class) so each pair has unique color & marker
    labels_source = np.array([f"source-{int(lbl)}" for lbl in y_source])
    labels_target = np.array([f"target-{int(lbl)}" for lbl in y_target])
    combined_labels = np.concatenate([labels_source, labels_target])

    # Unique categories, palette and markers mapping
    unique_cats = list(dict.fromkeys(combined_labels.tolist()))
    base_palette = sns.color_palette("tab10", n_colors=max(10, len(unique_cats)))
    palette_map = {cat: base_palette[idx % len(base_palette)] for idx, cat in enumerate(unique_cats)}
    base_markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    markers_map = {cat: base_markers[idx % len(base_markers)] for idx, cat in enumerate(unique_cats)}

    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_combined[:, 0],
        y=X_combined[:, 1],
        hue=combined_labels,
        style=combined_labels,
        palette=palette_map,
        markers=markers_map,
        alpha=0.7,
        s=20,
        legend=True,
    )
    plt.title(f"PCA Plot - source vs target {split}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Domain-Class", loc="best")
    plt.tight_layout()
    save_dir = os.path.join(results_dir,
                             "combined", f"{split}/")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "pca.png"))

    plt.close()

def main(source_dir, target_dir, checkpoint_dir="./checkpoint/", results_dir="./results/", splits=["train", "val", "test"]):
    for split in splits:
        X_source, y_source = load_data(source_dir, split=split)
        X_target, y_target = load_data(target_dir, split=split)

        pca = get_pca(source_dir, 2, checkpoint_dir, split)
        get_pca_plot(X_source, y_source, pca, split, "source", results_dir)
        get_pca_plot(X_target, y_target, pca, split, "target", results_dir)
        
        get_joint_pca_plot(X_source, y_source, X_target, y_target, pca, split, results_dir)

