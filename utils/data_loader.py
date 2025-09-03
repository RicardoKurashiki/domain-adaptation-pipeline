import os
import numpy as np

def main(path, split="train"):
    X = np.load(os.path.join(path, split, "vectors.npy"))
    y = np.load(os.path.join(path, split, "labels.npy"))
    return X, y