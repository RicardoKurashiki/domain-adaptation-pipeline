import numpy as np

def main(X, y):
    """
	Separate data into dictionaries based on their labels.
	Args:
		X (np.array): Array of feature vectors.
		y (np.array): Array of corresponding labels.
	Returns:
		dict: Dictionary where keys are class labels (int) and values
			  are data subsets belonging to that class.
	"""
    result = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        result[int(label)] = X[np.array(y) == label]
    return result