from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

from utils import load_data
from ops import get_pca

import tensorflow as tf

def get_pca_plot(X, y, pca, split, results_dir):
    X_pca = pca.transform(X)
    
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis")
    plt.title(f"PCA Plot - {split}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class", loc="best")
    plt.tight_layout()
    save_dir = os.path.join(results_dir, "pca/")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "pca.png"))
    
    plt.close()

def check_classification(model, data, labels, split_name="mapped", title="[MODEL] Data Classification", output_path="./results/"):
	output_path = os.path.join(output_path, "classification/")
	os.makedirs(output_path, exist_ok=True)
	feature_classifier = tf.keras.models.load_model(model)
	
	prediction = feature_classifier.predict(data)
	prediction = np.argmax(prediction, axis=1)

	# Classification Report
	report = classification_report(labels, prediction, output_dict=True)
	df_report = pd.DataFrame(report).transpose()
	df_report = df_report.round(3)
	df_report.to_csv(os.path.join(output_path, f"{split_name}_classification_report.csv"), index=True)

	# Confusion Matrix
	cm = confusion_matrix(labels, prediction)
	fig = plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 14})
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.title(title)
	fig.savefig(os.path.join(output_path, f"{split_name}_confusion_matrix.png"))
	plt.close()

def main(source_dir, data_dir, model_dir, checkpoint_dir="./checkpoint/", results_dir="./results/"):
    X, y = load_data(data_dir, "mapped")
    pca = get_pca(source_dir, 2, checkpoint_dir)
    get_pca_plot(X, y, pca, "mapped", results_dir)
    check_classification(model_dir, X, y, "mapped", "[MODEL] Data Classification", results_dir)