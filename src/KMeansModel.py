from sklearn.cluster import KMeans

class KMeansModel:
    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters
        self.model = KMeans(n_clusters=num_of_clusters)


    def fit_data(self):
        # Call fit() function from sklearn
        pass

    def predict(self):
        pass

    def get_cluster_analysis(self):
        pass
