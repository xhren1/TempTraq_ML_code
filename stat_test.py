from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics as metrics
from tslearn.clustering import silhouette_score
from tslearn.clustering import TimeSeriesKMeans


def compute_silhouette_and_accuracy(seed, temp_array, true_label, cluster_number):
    """
    Compute the silhouette score and accuracy for the clustering model and random generating labels.

    Parameters:
    seed (int): The random seed.
    temp_array (np.array): The input data for clustering.
    true_label (np.array): The binary labels for fever causes.
    cluster_number (int): The number of clusters.

    Returns:
    model_silhouette_score (float): The silhouette score of the clustering model.
    random_silhouette_score (float): The silhouette score of the random generating labels.
    model_acc (float): The accuracy of the clustering model.
    random_acc (float): The accuracy of the random generating labels.
    """
    np.random.seed(seed)
    km = TimeSeriesKMeans(n_clusters=cluster_number,
                        n_init=2,
                        metric="dtw",
                        verbose=False,
                        max_iter_barycenter=10,
                        random_state=seed)
    prediect_result = km.fit_predict(temp_array)
    model_silhouette_score = silhouette_score(temp_array, prediect_result, metric="dtw")

    mapped_labels = np.zeros_like(prediect_result)
    for i in range(cluster_number):
        mapped_labels[prediect_result == i] = np.bincount(true_label[prediect_result == i]).argmax()
    model_acc = metrics.accuracy_score(true_label, mapped_labels)
    
    random_labels = np.random.randint(0, cluster_number, 30)
    random_silhouette_score = silhouette_score(temp_array, random_labels, metric="dtw")
    mapped_random_labels = np.zeros_like(random_labels)
    for i in range(cluster_number):
        mapped_random_labels[random_labels == i] = np.bincount(true_label[random_labels == i]).argmax()
    random_acc = metrics.accuracy_score(true_label, mapped_random_labels)
    
    return model_silhouette_score,random_silhouette_score,model_acc, random_acc