import os
import numpy as np
import tensorflow as tf
from modules.cluster_ae import cluster_ae

from utils import load_data
from ops import load_clusters

def map_data_to_centroids(X_data, y_labels, centroids_dict):
    mapped_targets = []
    for i, x_vec in enumerate(X_data):
        label = y_labels[i]
        class_centroids = centroids_dict.get(str(label))
            
        distances = [np.linalg.norm(x_vec - c) for c in class_centroids]
        closest_centroid_idx = np.argmin(distances)
        mapped_targets.append(class_centroids[closest_centroid_idx])
        
    return np.array(mapped_targets)

def get_callbacks():
    return [
       tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=10,
			mode="min",
			restore_best_weights=True,
			verbose=1
		)
    ]

def main(source_dir, target_dir, checkpoint_dir="./checkpoint/", results_dir="./results/", epochs=100, batch_size=64):
    clusters = load_clusters(checkpoint_dir, "train")
    if clusters is None:
        raise ValueError("Clusters not found")
    
    X_train, y_train = load_data(target_dir, "train")
    X_val, y_val = load_data(target_dir, "val")
    X_test, y_test = load_data(target_dir, "test")

    y_train_mapped = map_data_to_centroids(X_train, y_train, clusters)
    y_val_mapped = map_data_to_centroids(X_val, y_val, clusters)

    input_dim = X_train.shape[1]

    model = cluster_ae(input_dim=input_dim)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        X_train,
        y_train_mapped,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_mapped),
        callbacks=get_callbacks()
    )
    
    save_dir = os.path.join(results_dir, "cluster_ae/")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "cluster_ae.keras"))
    np.save(os.path.join(save_dir, "history.npy"), history.history)

    # Mapping Test Data to Centroids
    mapped_X_test = model.predict(X_test)
    mapped_dir = os.path.join(save_dir, "mapped/")
    os.makedirs(mapped_dir, exist_ok=True)
    np.save(os.path.join(mapped_dir, "vectors.npy"), mapped_X_test)
    np.save(os.path.join(mapped_dir, "labels.npy"), y_test)

    return save_dir