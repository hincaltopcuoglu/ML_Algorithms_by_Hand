import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # Regularization parameter
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Convergence tolerance
        self.coef_ = None  # Coefficients
        self.intercept_ = 0.0  # Intercept
        self.X_mean_ = None  # For centering at prediction time
        self.X_std_ = None

    def fit(self, X, y):
        # Normalize and center features
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        X_std_mask = self.X_std_ == 0
        self.X_std_[X_std_mask] = 1  # avoid divide by zero
        X_normed = (X - self.X_mean_) / self.X_std_

        y_mean = y.mean()
        y_centered = y - y_mean

        self.coef_ = np.zeros(X.shape[1])
        n = len(y)

        for iteration in range(self.max_iter):
            prev_coef = self.coef_.copy()

            for j in range(len(self.coef_)):
                r = y_centered - X_normed @ self.coef_ + self.coef_[j] * X_normed[:, j]
                rho = (1 / n) * np.dot(X_normed[:, j], r)
                self.coef_[j] = self._soft_threshold(rho, self.alpha)

            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break

        self.intercept_ = y_mean - np.dot((self.X_mean_ / self.X_std_), self.coef_)

        if np.isnan(self.coef_).any():
            print("Warning: NaNs in coefficients!")


    def _soft_threshold(self, x, alpha):
        if x > alpha:
            return x - alpha
        elif x < -alpha:
            return x + alpha
        else:
            return 0.0

    def predict(self, X):
        X_normed = (X - self.X_mean_) / self.X_std_
        return X_normed @ self.coef_ + self.intercept_

