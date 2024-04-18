import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

red_wine_data = pd.read_csv('winequality-red.csv', sep=';')
white_wine_data = pd.read_csv('winequality-white.csv', sep=';')

def predict_wine_quality(n_clusters, new_data_str, wine_type):
    red_features = red_wine_data.drop(columns=['quality'])
    white_features = white_wine_data.drop(columns=['quality'])

    # Parse inputted data string
    new_data = np.fromstring(new_data_str, dtype=float, sep=';')
    new_data_df = pd.DataFrame([new_data], columns=red_features.columns)

    #KMeans clustering for red wine
    kmeans_red = KMeans(n_clusters=n_clusters)
    red_labels = kmeans_red.fit_predict(red_features)

    #KMeans clustering for white wine
    kmeans_white = KMeans(n_clusters=n_clusters)
    white_labels = kmeans_white.fit_predict(white_features)

    if wine_type == 'red':
        cluster_label = kmeans_red.predict(new_data_df)[0]
        average_quality = red_wine_data.loc[red_labels == cluster_label, 'quality'].mean()
    elif wine_type == 'white':
        cluster_label = kmeans_white.predict(new_data_df)[0]
        average_quality = white_wine_data.loc[white_labels == cluster_label, 'quality'].mean()
    else:
        print("Invalid wine type. Please enter 'red' or 'white'.")
        return None

    return average_quality


def main():
    while True:
        n_clusters = int(input("Enter the number of clusters: "))

        new_data_str = input("Enter new data string (semicolon-separated values without quality): ")

        wine_type = input("Enter the type of wine (red or white): ").lower()

        # Predict wine quality
        average_quality = predict_wine_quality(n_clusters, new_data_str, wine_type)
        if average_quality is not None:
            print(f"The predicted average quality of the wine is: {average_quality:.2f}")

if __name__ == "__main__":
    main()
