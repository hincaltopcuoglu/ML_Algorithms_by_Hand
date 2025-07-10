import numpy as np
from collections import Counter



class KNeighboursClassifier:
    """
    A K-Nearest Neighbors classifier implemented from scratch using NumPy.
    Supports uniform and distance-based weighting.
    """

    def __init__(self, k=3, weights='uniform'):
        """
        Initializes the classifier.
        :param k: Number of neighbors to use.
        :param weights: 'uniform' or 'distance'.
                        'uniform': All neighbors have equal vote.
                        'distance': Closer neighbors have a stronger vote.
        """
        assert k >= 1, "k must be a positive integer"
        assert weights in ['uniform', 'distance'], "Weights must be 'uniform' or 'distance'."
        self.k = k
        self.weights = weights
        
    def fit(self, X, y):
        """
        "Trains" the classifier by memorizing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts class labels for a set of new data points.
        This version is clear, readable, and correct.
        """
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        """
        A private helper method to predict the label for a single data point (x).
        """
        # 1. Calculate Euclidean distances
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

        # 2. Get the indices and labels of the k-nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        if self.weights == 'uniform':
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]

        elif self.weights == 'distance':
            # Get the distances of only the k-nearest neighbors
            k_nearest_distances = distances[k_nearest_indices]

            # Dictionary to store the sum of weights for each class label
            weighted_votes = {}
            
            # This loop iterates through the k-nearest neighbors
            for i, label in enumerate(k_nearest_labels):
                distance = k_nearest_distances[i]
                
                # Calculate weight (inverse distance). Add a small epsilon to avoid division by zero.
                weight = 1 / (distance + 1e-9)

                # Add the weight to the corresponding label's total in the dictionary
                if label not in weighted_votes:
                    weighted_votes[label] = 0
                weighted_votes[label] += weight
            
            # Return the label (key) with the maximum total weight (value)
            return max(weighted_votes, key=weighted_votes.get)