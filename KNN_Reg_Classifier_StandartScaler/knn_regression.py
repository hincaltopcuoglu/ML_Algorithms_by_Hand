import numpy as np

class KNNRegressor:
    """
    K-Nearest Neighbors regressor from scratch.
    """

    def __init__(self, k=3):
        """
        Initializes the KNN Regressor.

        Args:
            k (int): The number of nearest neighbors to consider for prediction.
        """
        self.k = k

    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_new):
        """
        Predicts the target values for new data points.

        Args:
            X_new (np.ndarray): New data to predict, shape (n_test_samples, n_features).

        Returns:
            np.ndarray: The predicted values for each sample in X_new.
        """

        predictions = [self._predict_single(x_sample) for x_sample in X_new]

        return np.array(predictions)




    
    def _predict_single(self, x):
        """
        A helper method to predict the value for a single data point.

        Args:
            x (np.ndarray): A single data point of shape (n_features,).

        Returns:
            float: The predicted value for the single data point.
        """

        distances = [np.sqrt(np.sum((x - p)**2)) for p in self.X_train]

        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:self.k]

        prediction =  (1 / self.k) * np.sum(self.y_train[k_nearest_indices])

        return prediction