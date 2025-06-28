# =============================================================================
# Step 1: Imports
# =============================================================================
import numpy as np
import time # Import the time module

# Tools for data handling and preparation
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Tools for modeling and evaluation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Import your custom-built algorithm
from ridge_regression import RidgeRegression


# =============================================================================
# Step 2: Define Hyperparameters
# =============================================================================
ALPHA = 0.1
LEARNING_RATE = 0.01
# Let's set this to a high number to ensure convergence
# Feel free to change this back to 1000 to see the speed difference
N_ITERS = 10000 
RANDOM_STATE = 42 


# =============================================================================
# Step 3: Create and Prepare the Dataset
# =============================================================================
print("1. Preparing Data...")
X, y = make_regression(
    n_samples=500, 
    n_features=10, 
    noise=20, 
    random_state=RANDOM_STATE
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.\n")


# =============================================================================
# Step 4: Train and Evaluate Custom Model
# =============================================================================
print("--- Training My Custom Ridge Model ---")

my_ridge = RidgeRegression(
    learning_rate=LEARNING_RATE, 
    n_iters=N_ITERS, 
    alpha=ALPHA
)

# Record time before training
start_time_my_model = time.time()
my_ridge.fit(X_train_scaled, y_train)
# Record time after training
end_time_my_model = time.time()
duration_my_model = end_time_my_model - start_time_my_model

my_predictions = my_ridge.predict(X_test_scaled)
my_mse = mean_squared_error(y_test, my_predictions)

print(f"Training Time: {duration_my_model:.6f} seconds")
print(f"Learned Bias (b): {my_ridge.bias:.4f}")
print(f"Learned Weights (w): {my_ridge.weights}")
print(f"Mean Squared Error (MSE): {my_mse:.4f}")
print("-" * 35 + "\n")


# =============================================================================
# Step 5: Train and Evaluate Scikit-learn's Model
# =============================================================================
print("--- Training Scikit-learn's Ridge Model ---")

sklearn_ridge = Ridge(alpha=ALPHA)

# Record time before training
start_time_sklearn = time.time()
sklearn_ridge.fit(X_train_scaled, y_train)
# Record time after training
end_time_sklearn = time.time()
duration_sklearn = end_time_sklearn - start_time_sklearn

sklearn_predictions = sklearn_ridge.predict(X_test_scaled)
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

print(f"Training Time: {duration_sklearn:.6f} seconds")
print(f"Learned Bias (b): {sklearn_ridge.intercept_:.4f}")
print(f"Learned Weights (w): {sklearn_ridge.coef_}")
print(f"Mean Squared Error (MSE): {sklearn_mse:.4f}")
print("-" * 35 + "\n")


# =============================================================================
# Step 6: Final Comparison
# =============================================================================
print("--- Runtime Comparison Summary ---")
print(f"My Model ({N_ITERS} iterations): {duration_my_model:.6f} seconds")
print(f"Scikit-learn Model:      {duration_sklearn:.6f} seconds")