import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.

    Attributes:
        lr (float): The learning rate for gradient descent.
        n_iters (int): The number of iterations for gradient descent.
        weights (np.array): The learned weights for each feature after training.
        bias (float): The learned bias term after training.
        cost_history (list): A history of the cost function value during training.
                             Useful for debugging and visualization.
    """

    def __init__(self,learning_rate=0.001, n_iterations=1000):
        """
        The constructor for the LogisticRegression class.
        Initializes the hyperparameters.
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.cost_history = []
        X = np.array(X)
        y = np.array(y)


        # m: The number of training samples.
        # n: The number of features.

        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        # this is the training loop
        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            A = self._sigmoid(z)
            # To prevent log(0) errors, we clip the values of A.
            epsilon = 1e-15  # A very small number to avoid log(0)
            A_clipped = np.clip(A, epsilon, 1 - epsilon)
            cost_func = -(1/m) * np.sum(y * np.log(A_clipped) + (1- y) * np.log(1 - A_clipped))
            self.cost_history.append(cost_func)
            dw = (1/m) * np.dot(X.T, (A - y))
            db = (1/m) * np.sum(A - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db




    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X (np.array): New data to predict on, shape (n_samples, n_features).

        Returns:
            np.array: An array of predicted class labels (0 or 1).
        """
        # calculate the predictions using the learned parameters
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        predicted_class = (A >= 0.5).astype(int)

        return predicted_class


    def _sigmoid(self, z):
        A = 1 / (1 + np.exp(-z))
        return A
    
