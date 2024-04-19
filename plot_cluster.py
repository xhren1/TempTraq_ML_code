import pandas as pd
import numpy as np
from tslearn.clustering import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt


def data_plot(train_data,label):
    '''
    Plot the data before clustering.
    
    Args:
        train_data (np.array): preprocessed data for clustering
        orignial_percent (list): the percentage of original data

    Returns:
        None, but plot the data
    '''

    x = np.arange(-train_data.shape[1],train_data.shape[1],2)
    for i in range(len(train_data)):
        
        plt.subplot(int(len(train_data)/5)+1, 5, i + 1)
        plt.plot(x,train_data[i].ravel(), "k-", alpha=.5)
        plt.axhline(y=38, color='red', linestyle='--',lw=0.8)
        plt.axvline(x=0, color='green', linestyle='--',lw=0.8)
        plt.xlim(-train_data.shape[1], train_data.shape[1])
        plt.ylim(35, 41)
        plt.title(label[i])


def cluster_number_decision(train_data, min_clusters=2,max_clusters=9,seed = 10):
    '''
    Plot the square distances for different number of clusters.
    
    Args:
        train_data (np.array): preprocessed data for clustering
        min_clusters (int): minium number of clusters
        max_clusters (int): maximum number of clusters
        
    Returns:
        None, only plot the square distances for different number of clusters
    '''
    squared_distances = []
    for n_clusters in range(min_clusters,max_clusters+1):
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_init=2, max_iter_barycenter=10, verbose=False, random_state=seed)
        km = km.fit(train_data)
        squared_distances.append(km.inertia_)
    plt.plot(range(min_clusters,max_clusters+1),squared_distances,"*-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Squared distances")
    
    
def DTW_KMeans_clustering(train_data, cluster_number,seed = 10):
    '''
    Cluster the data using DTW KMeans. 
    Plot the cluster centers and the data for each cluster.
    Print the silhouette score for the clustering.
    
    Args:
        train_data (np.array): preprocessed data for clustering
        cluster_number (int): number of clusters
        seed (int): random seed
        
    Returns:
        prediect_result (np.array): the cluster labels for each data point
        km.cluster_centers_ (np.array): the cluster centers
        km (TimeSeriesKMeans): the clustering model
    '''
    
    km = TimeSeriesKMeans(n_clusters=cluster_number,
                            n_init=2,
                            metric="dtw",
                            verbose=False,
                            max_iter_barycenter=10,
                            random_state=seed)
    prediect_result = km.fit_predict(train_data)
    # print silhouette score
    silhouette = silhouette_score(train_data, prediect_result, metric="dtw")
    # print("silhouette score: {:.2f}".format(silhouette))
    
    x = np.arange(-train_data.shape[1],train_data.shape[1],2).reshape(-1,1)

    for i in range(cluster_number):
        plt.subplot(3, cluster_number, i+1)
        # plot all the patient temperature curves
        for j in train_data[prediect_result == i]:
            plt.plot(x, j.ravel(), "k-", alpha=.25)
        # plot the cluster center in blue
        plt.plot(x, km.cluster_centers_[i].ravel(), "blue")
        plt.xlim(-train_data.shape[1], train_data.shape[1])
        # set the y axis limit to body temperature range
        plt.ylim(34, 41)
        plt.axhline(y=38, color='red', linestyle='--',lw=0.8)
        plt.axvline(x=0, color='green', linestyle='--',lw=0.8)
        
        plt.text(0.55, 0.85,'Cluster %d' % (i + 1),
                transform=plt.gca().transAxes, size=14)

        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()

    return prediect_result, km.cluster_centers_, km, silhouette