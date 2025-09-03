import os
import pickle as pk
from utils import load_data

from sklearn.decomposition import PCA

def main(data_dir, n_components=2, checkpoint_dir="./checkpoint/", split="train"):
    save_path = os.path.join(checkpoint_dir, "pca", f"{split}/")
    if os.path.exists(save_path):
        pca = pk.load(open(os.path.join(save_path, "pca.pkl"), "rb"))
        return pca
    X, _ = load_data(data_dir, split=split)
    pca = PCA(n_components=n_components)
    X_fit = pca.fit_transform(X)
    os.makedirs(save_path, exist_ok=True)
    pk.dump(pca, open(os.path.join(save_path, "pca.pkl"), "wb"))
    return pca