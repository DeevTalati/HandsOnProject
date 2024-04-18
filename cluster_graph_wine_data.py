import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.widgets import CheckButtons

red_wine_data = pd.read_csv('winequality-red.csv', sep=';')
white_wine_data = pd.read_csv('winequality-white.csv', sep=';')

def calculate_silhouette_score(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg

def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters+1):
        silhouette_scores.append(calculate_silhouette_score(data, n_clusters))
    return silhouette_scores.index(max(silhouette_scores)) + 2

def visualize_clusters(red_data, white_data, x_axis, y_axis, n_clusters):
    red_features = red_data[[x_axis, y_axis]]
    white_features = white_data[[x_axis, y_axis]]

    red_optimal_clusters = find_optimal_clusters(red_features)
    white_optimal_clusters = find_optimal_clusters(white_features)

    print("Red Wine - Number of clusters:", red_optimal_clusters)
    print("White Wine - Number of clusters:", white_optimal_clusters)

    #KMeans clustering for red wine
    kmeans_red = KMeans(n_clusters=red_optimal_clusters)
    kmeans_red.fit(red_features)
    cluster_labels_red = kmeans_red.predict(red_features)
    centroids_red = kmeans_red.cluster_centers_

    #KMeans clustering for white wine
    kmeans_white = KMeans(n_clusters=white_optimal_clusters)
    kmeans_white.fit(white_features)
    cluster_labels_white = kmeans_white.predict(white_features)
    centroids_white = kmeans_white.cluster_centers_

    fig, ax = plt.subplots(figsize=(19, 10))

    # Scatter plot for red wine
    scatter_red = ax.scatter(red_features[x_axis], red_features[y_axis], c=cluster_labels_red, cmap='viridis', s=50, alpha=0.5, marker='o', label='Red Wine')

    # Scatter plot for white wine
    scatter_white = ax.scatter(white_features[x_axis], white_features[y_axis], c=cluster_labels_white, cmap='viridis', s=50, alpha=0.5, marker='s', label='White Wine')

    # Centroids
    red_centroids_plot = ax.scatter(centroids_red[:, 0], centroids_red[:, 1], c='red', s=200, marker='x', label='Red Wine Centroids', visible=False)
    white_centroids_plot = ax.scatter(centroids_white[:, 0], centroids_white[:, 1], c='blue', s=200, marker='x', label='White Wine Centroids', visible=False)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('Clustering: {} vs {}'.format(x_axis, y_axis))
    plt.colorbar(scatter_red, label='Cluster')

    ax.text(0.05, -0.1, 'Red Wine: Red Centroid "X"  and Shape = Circle\nWhite Wine: Blue Centroid "X"  and Shape = Square',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    rax = plt.axes([0.83, 0.7, 0.18, 0.25])
    check = CheckButtons(rax, ('Red Wine', 'White Wine', 'Red Wine Centroids', 'White Wine Centroids (Blue)'), (True, True, False, False))

    def func(label):
        if label == 'Red Wine':
            scatter_red.set_visible(not scatter_red.get_visible())
        elif label == 'White Wine':
            scatter_white.set_visible(not scatter_white.get_visible())
        elif label == 'Red Wine Centroids':
            red_centroids_plot.set_visible(not red_centroids_plot.get_visible())
        elif label == 'White Wine Centroids (Blue)':
            white_centroids_plot.set_visible(not white_centroids_plot.get_visible())
        plt.draw()

    check.on_clicked(func)

    plt.show()

def main():
    while True:
        print("Available columns for selection:")
        print(red_wine_data.columns)

        x_axis = input("Enter the column name for the x-axis (or type 'exit' to quit): ")
        if x_axis.lower() == 'exit':
            break

        y_axis = input("Enter the column name for the y-axis: ")

        visualize_clusters(red_wine_data, white_wine_data, x_axis, y_axis, 0)

if __name__ == "__main__":
    main()