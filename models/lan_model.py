import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class LANDetector:
    def __init__(self, k=5):
        self.k = k

    def fit_predict(self, X):
        distances = euclidean_distances(X)
        scores = []
        for i in range(len(X)):
            neighbors = np.argsort(distances[i])[1:self.k+1]
            score = np.mean(distances[i][neighbors])
            scores.append(score)
        return np.array(scores)
