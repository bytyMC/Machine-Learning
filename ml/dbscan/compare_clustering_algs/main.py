import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def generate_data(n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4):
    """Generate sample data using make_blobs."""
    X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=0)
    return X, labels_true

def perform_clustering(X, method='dbscan', **kwargs):
    """Apply the specified clustering method."""
    if method == 'dbscan':
        model = DBSCAN(**kwargs)
    elif method == 'kmeans':
        model = KMeans(**kwargs)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(**kwargs)
    else:
        raise ValueError("Unknown clustering method: {}".format(method))
    
    labels = model.fit_predict(X)
    return labels

def evaluate_clustering(labels_true, labels, X):
    """Print various clustering evaluation metrics."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) if -1 in labels else 0

    print('Estimated number of clusters: %d' % n_clusters)
    print('Estimated number of noise points: %d' % n_noise)
    
    if n_clusters > 0:
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
        
        if n_clusters > 1:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        else:
            print("Silhouette Coefficient: Not applicable (only one cluster)")
    else:
        print("No valid clusters found. Cannot compute clustering metrics.")

def plot_clusters(X, labels, title):
    """Plot the clustered data."""
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8, 6))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=tuple(col), label=f'Cluster {k}' if k != -1 else 'Noise', s=30)

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(title + ".png")

def main():
    X, labels_true = generate_data()
    X = StandardScaler().fit_transform(X)

    # DBSCAN
    labels_dbscan = perform_clustering(X, method='dbscan', eps=0.3, min_samples=10)
    print("\nDBSCAN Results:")
    evaluate_clustering(labels_true, labels_dbscan, X)
    plot_clusters(X, labels_dbscan, 'DBSCAN Clustering')

    # K-Means
    labels_kmeans = perform_clustering(X, method='kmeans', n_clusters=3)
    print("\nK-Means Results:")
    evaluate_clustering(labels_true, labels_kmeans, X)
    plot_clusters(X, labels_kmeans, 'K-Means Clustering')

    # Agglomerative Clustering
    labels_agglo = perform_clustering(X, method='agglomerative', n_clusters=3)
    print("\nAgglomerative Clustering Results:")
    evaluate_clustering(labels_true, labels_agglo, X)
    plot_clusters(X, labels_agglo, 'Agglomerative Clustering')

if __name__ == "__main__":
    main()