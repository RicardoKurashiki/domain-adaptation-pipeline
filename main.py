import os

import argparse
import tensorflow as tf

from ops import get_clusters, get_pca
from pipelines import view_data

parser = argparse.ArgumentParser(prog="Domain Transfer Model Pipeline")

parser.add_argument('--dataset_name', default="chest-xray-processed", help='Name of the dataset. Defaults to %(default)s',
                    choices=["chest-xray-processed", "rsna-processed", "mnist", "mnist-m"])
parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs. Defaults to %(default)s')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training. Defaults to %(default)s')
parser.add_argument('--clusters', default=1, type=int, help='KMeans Clusters. Defaults to %(default)s')
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility. Defaults to %(default)s')

args = parser.parse_args()

def set_gpu_usage(seed=42):
    tf.random.set_seed(seed)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found, using CPU")

def main():
    seed = args.seed
    set_gpu_usage(seed=seed)
    dataset_name = args.dataset_name
    epochs = args.epochs
    batch_size = args.batch_size
    clusters = args.clusters
    source_dir = f"./datasets/{dataset_name}/source/"
    target_dir = f"./datasets/{dataset_name}/target/"
    model_dir = f"./models/{dataset_name}/feature_classifier.keras"
    checkpoint_dir = f"./checkpoint/{dataset_name}/"
    results_dir = f"./results/{dataset_name}/"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    get_clusters(source_dir, clusters, checkpoint_dir, seed)

    get_pca(source_dir, 2, checkpoint_dir, "train")
    get_pca(source_dir, 2, checkpoint_dir, "val")
    get_pca(source_dir, 2, checkpoint_dir, "test")

    view_data(source_dir, target_dir, checkpoint_dir, results_dir)

if __name__ == "__main__":
    main()