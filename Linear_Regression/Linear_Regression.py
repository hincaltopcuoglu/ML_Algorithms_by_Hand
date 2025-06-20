import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.coefficient = None
        self.intercept = None
        self.is_fitted = False
        self.fit_intercept = fit_intercept


    # Multiple Linear Regression Support
    def fit(self, X, y):
        """
        Fit the linear regression model using the normal equation.

        Args:
        X: Feature matrix (numpy array or convertible)
        y: Target vector (numpy array or convertible)

        Returns:
        self: Fitted model instance
        """
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # add columns of ones for intercept if needed
        if self.fit_intercept:
            X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        else:
            X_with_intercept = X

        # Solve for coefficients using normal equation: β = (X^T X)^(-1) X^T y
        #beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        # Use pseudo-inverse instead of inverse
        beta = np.linalg.pinv(X_with_intercept) @ y

        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.intercept = 0
            self.coefficient = beta

        self.is_fitted = True

        return self
    
    def predict(self, X):
        """
        Predict target values using the fitted model.

        Args:
        X: Feature matrix (numpy array or convertible)

        Returns:
        y_pred: Predicted target values (numpy array)
        """
        # Check if model is fitted, raise error if not
        # Convert X to numpy array if needed
        # Calculate predictions: slope * X + intercept
        # Return predictions

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first")
        
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1,1) # ensure 2d matrix multiplication


        y_pred = X @ self.coefficient + self.intercept
        
        return y_pred
    
    # Add Regularization Options
    def fit_ridge(self, X, y, alpha=1.0):
        """
        Fit the ridge regression model (L2 regularization).

        Args:
        X: Feature matrix (numpy array or convertible)
        y: Target vector (numpy array or convertible)
        alpha: Regularization strength (float)

        Returns:
        self: Fitted model instance
        """
        X = np.array(X)
        y = np.array(y)

        # add column of ones for intercept
        if self.fit_intercept:
            X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        else:
            X_with_intercept = X

        # identity matrix for regularization
        n_features = X_with_intercept.shape[1]
        identity = np.identity(n_features)
        if self.fit_intercept:
            identity[0,0] = 0 # don't regularize intercept

        # Ridge regression formula: β = (X^T X + αI)^(-1) X^T y
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + alpha * identity) @ (X_with_intercept @ y)

        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.intercept = 0
            self.coefficient = beta

        self.is_fitted = True

        return self
    
    def score(self, X, y):
        # Check if model is fitted, raise error if not
        # Get predictions for X
        # Calculate mean of true y values
        # Calculate total sum of squares: sum of (y - y_mean) squared
        # Calculate residual sum of squares: sum of (y - predictions) squared
        # Calculate and return R² score: 1 - (residual_sum / total_sum)

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first")
        
        y = np.array(y)

        # get predictions using predict method
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y-y_mean)**2)
        if ss_total == 0:
            # All y values are the same, R² is not defined; return 0 or 1 depending on convention
            return 0.0
        ss_residual = np.sum((y-y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)

        return r2
    
    # Model Evaluation Methods
    def mse(self, X, y):
        """Calculate Mean Squarred Error"""
        y_pred = self.predict(X)
        y = np.array(y)
        
        return np.mean((y-y_pred)**2)
    
    def mae(self, X, y):
        """Calculate Mean Absolute Error"""
        y_pred = self.predict(X)
        y = np.array(y)
        
        return np.mean(np.abs(y-y_pred))
    
    # Add Feature Importance and Model Summary
    def summary(self):
        """Print Model Summary Statistics"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.Call fit() first")

        print("Model Summary")
        print("-" * 50)
        if self.fit_intercept:
            print(f"Intercept: {self.fit_intercept:.4f}")

        print("Coefficients")
        for i, coef in enumerate(self.coefficient):
            print(f" Feature {i+1}: {coef:.4f}")

        print("-" * 50)       

    # Cross-Validation Support
    def cross_validate(self, X, y, k=5, time_based=False, random_state= None):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)

        # create k-folds
        if random_state is not None:
            np.random.seed(random_state)

        scores = []
        if time_based:
            fold_size = n_samples // k
            for i in range(1,k+1):
                train_end = fold_size * i
                test_start = train_end
                test_end = min(test_start + fold_size, n_samples)

                if test_start >= n_samples:
                    break # no more test data

                X_train = X[:train_end]
                y_train = y[:train_end]
                X_test = X[test_start:test_end]
                y_test = y[test_start:test_end]

                self.fit(X_train, y_train)
                scores.append(self.score(X_test,y_test))
        else:
            indices = np.random.permutation(n_samples)
            fold_size = n_samples // k
            for i in range(k):
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < k -1 else n_samples

                test_indices = indices[test_start:test_end]
                train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]

                self.fit(X_train, y_train)
                scores.append(self.score(X_test, y_test))

        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    def plot_residuals(self, X, y):
        """Plot residuals to check model assumptions"""
        y_pred = self.predict(X)
        residuals = y - y_pred

        plt.figure(figsize=(10,6))
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual plot')
        plt. grid(True)
        plt.show()

        # also show histogram of residuals
        plt.figure(figsize=(10,6))
        plt.hist(residuals, bins=20)
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.grid(True)
        plt.show()

    def preprocess_features(self, X, standardize=False):
        "Preprocess features before fitting or prediction"
        X = np.array(X)

        if standardize:
            if not hasattr(self, 'feature_mean_') or not hasattr(self, 'feature_std_'):
                self.feature_mean_ = np.mean(X,axis=0)
                self.feature_std_ = np.std(X, axis=0)

            # avoıd division by zero
            self.feature_std_ = np.where(self.feature_std_==0,1,self.feature_std_)

            return (X - self.feature_mean_) / self.feature_std_
        
        return X
    



# Seed for reproducibility
np.random.seed(42)

# Generate complex dataset
n_samples = 200
X1 = np.random.uniform(0, 10, n_samples)
X2 = np.random.normal(5, 2, n_samples)
X3 = np.random.binomial(1, 0.3, n_samples)  # binary feature
X = np.column_stack((X1, X2, X3))

# Target with linear + nonlinear terms + noise
y = 3.5 * X1 - 2.2 * X2 + 4.7 * X3 + 0.5 * (X1 ** 2) + np.random.normal(0, 3, n_samples)

# Split into train/test
split_idx = int(n_samples * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Create model instance
model = LinearRegression(fit_intercept=True)

# 1. Fit model
model.fit(X_train, y_train)

# 2. Print summary
model.summary()

# 3. Predict on test set
y_pred = model.predict(X_test)
print("Predictions on test data:", y_pred[:5])

# 4. Score on train and test
print(f"R² on training data: {model.score(X_train, y_train):.4f}")
print(f"R² on test data: {model.score(X_test, y_test):.4f}")

# 5. Calculate MSE and MAE on test data
print(f"MSE on test data: {model.mse(X_test, y_test):.4f}")
print(f"MAE on test data: {model.mae(X_test, y_test):.4f}")

# 6. Plot residuals on test data
model.plot_residuals(X_test, y_test)

# 7. Preprocess features (standardize) on train and test
X_train_std = model.preprocess_features(X_train, standardize=True)
X_test_std = model.preprocess_features(X_test, standardize=True)
print("First 5 standardized training features:\n", X_train_std[:5])

# 8. Cross-validation (random split)
cv_results = model.cross_validate(X, y, k=5, time_based=False, random_state=42)
print("Cross-validation results (random split):", cv_results)

# 9. Cross-validation (time-based split)
cv_results_time = model.cross_validate(X, y, k=5, time_based=True)
print("Cross-validation results (time-based split):", cv_results_time)