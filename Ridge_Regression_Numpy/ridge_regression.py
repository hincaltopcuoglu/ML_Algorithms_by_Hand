import numpy as np

class RidgeRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, alpha=1.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.alpha = alpha
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # gradient for the weights
            y_predicted = np.dot(X, self.weights) + self.bias
            mse_gradient = (2 / n_samples) * np.dot(X.T,(y_predicted - y))
            regularization_gradient = 2 * self.alpha * self.weights
            dw = mse_gradient + regularization_gradient

            # gradient for the bias
            db = (2 / n_samples) *  np.sum(y_predicted - y)

            # update parameters
            # update the weigts using the dw 
            self.weights = self.weights - self.lr * dw
            
            #update bias using db gradient
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        # calculate the predictions using the learned parameters
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted