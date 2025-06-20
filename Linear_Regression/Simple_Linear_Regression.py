"""
Simple Linear Regression Class - Concept Guide
Core Concept:
Create a class that implements linear regression from scratch, following the equation y = mx + b (or y = slope * x + intercept).

Class Structure:

1-) Attributes to include:
slope: The coefficient (m) in the linear equation
intercept: The y-intercept (b) in the linear equation
is_fitted: Boolean flag to track if the model has been trained

2-)Methods to implement:
__init__:
Purpose: Initialize the model with default parameters
Parameters: None
Implementation: Set slope and intercept to None, is_fitted to False

fit:
Purpose: Calculate the slope and intercept using the formula
Parameters: X (features), y (target values)
Implementation steps:
Calculate means of X and y
Calculate slope using the formula:
slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)²)
Calculate intercept using:
intercept = y_mean - slope * x_mean
Set is_fitted to True
Return self (for method chaining)

predict:
Purpose: Make predictions using the fitted model
Parameters: X (features to predict on)
Implementation:
Check if model is fitted
Calculate y = slope * X + intercept
Return predictions

score:
Purpose: Calculate R² score (coefficient of determination)
Parameters: X (features), y (true target values)
Implementation steps:
Get predictions using predict method
Calculate total sum of squares: sum((y - y_mean)²)
Calculate residual sum of squares: sum((y - y_pred)²)
Calculate R²: 1 - (residual_sum / total_sum)

Mathematical Formulas:
Slope (m):
m = Σ((x_i - x̄) * (y_i - ȳ)) / Σ((x_i - x̄)²)
Intercept (b):
b = ȳ - m * x̄
R² Score:
R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
where ŷ_i are the predicted values
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.is_fitted = False

    def fit(self, X, y):
        # Convert inputs to numpy arrays if needed
        # Calculate means of X and y
        # Calculate numerator: sum of (x - x_mean) * (y - y_mean)
        # Calculate denominator: sum of (x - x_mean) squared
        # Calculate slope: numerator / denominator
        # Calculate intercept: y_mean - slope * x_mean
        # Set is_fitted to True
        # Return self for method chaining

        X = np.array(X).flatten() # ensure X is 1D
        y = np.array(y)

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y-y_mean))
        denominator = np.sum((X - x_mean)**2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        self.is_fitted = True

        return self


    def predict(self, X):
        # Check if model is fitted, raise error if not
        # Convert X to numpy array if needed
        # Calculate predictions: slope * X + intercept
        # Return predictions

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first")
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # ensure X is 1D or flatten if needed
        if X.ndim > 1:
            X = X.flatten()
        
        predictions = self.slope * X + self.intercept

        return predictions


    def score(self, X, y):
        # Check if model is fitted, raise error if not
        # Get predictions for X
        # Calculate mean of true y values
        # Calculate total sum of squares: sum of (y - y_mean) squared
        # Calculate residual sum of squares: sum of (y - predictions) squared
        # Calculate and return R² score: 1 - (residual_sum / total_sum)

        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first")
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # get predictions using predict method
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y-y_mean)**2)
        ss_residual = np.sum((y-y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)

        return r2
    



# create some sample data
X = np.linspace(0,10,100)
y = 2 * X + 1 + np.random.normal(0,1,100) # y=2x +1 + noise

# split into train and test sets
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# create and fit your model
model = SimpleLinearRegression()
model.fit(X_train,y_train)

# make predictions
y_pred = model.predict(X_test)

# Calculate and print R² score
r2 = model.score(X_test,y_test)
print(f"R² Score: {r2:.4f}")

# calculate score manually for verification
r2_manual = r2_score(y_test,y_pred)
print(f"R² Score (calculated manually): {r2_manual:.4f}")


# Visualize results
plt.figure(figsize=(10,6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label="Training Data")
plt.scatter(X_test, y_test, color= 'green', alpha=0.5, label="Testing Data" )
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predictions")
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()