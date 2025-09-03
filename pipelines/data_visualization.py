import os
from utils import load_data
from ops import get_pca

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

def main(source_dir, target_dir, checkpoint_dir="./checkpoint/", results_dir="./results/", splits=["train", "val", "test"]):
    for split in splits:
        X_source, y_source = load_data(source_dir, split=split)
        X_target, y_target = load_data(target_dir, split=split)

        pca = get_pca(source_dir, 2, checkpoint_dir, split)
        get_pca_plot(X_source, y_source, pca, split, "source", results_dir)
        get_pca_plot(X_target, y_target, pca, split, "target", results_dir)

