import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # Used only for generating sample data
from gradient_descent import GradientDescent



# ===================================================================
# NEW PROBLEM-SPECIFIC FUNCTIONS (For Logistic Regression)
# ===================================================================

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def log_loss(X, y, w):
    """Log Loss (Binary Cross-Entropy) function."""
    m = len(y)
    z = X @ w
    p = sigmoid(z)
    
    # Add a small epsilon to prevent log(0) which is -inf
    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)
    
    return - (1 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def log_loss_gradient(X, y, w):
    """Gradient of the Log Loss function."""
    m = len(y)
    z = X @ w
    p = sigmoid(z) # Predicted probabilities
    error = p - y
    return (1 / m) * X.T @ error

# ===================================================================
# MAIN EXECUTION BLOCK (The Test)
# ===================================================================
if __name__ == "__main__":
    # --- Step 1: Generate synthetic classification data ---
    print("--- 1. Generating Data ---")
    # Create two distinct blobs of data points for binary classification
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42, cluster_std=1.5)
    
    # Add the bias term (x0=1) to the feature matrix
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # --- Step 2: Initialize and run the optimizer ---
    print("\n--- 2. Running Optimizer for Logistic Regression ---")
    # Note we might need a higher learning rate or more iterations for this problem
    optimizer = GradientDescent(loss_fn=log_loss, grad_fn=log_loss_gradient, lr=0.5, max_iter=3000)
    
    # Start with a random guess for the weights
    initial_w = np.random.randn(X_b.shape[1])
    
    # Run the optimization!
    optimizer.optimize(X_b, y, initial_w)
    
    # --- Step 3: Visualize the results (Decision Boundary) ---
    print("\n--- 3. Generating Plots ---")
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Loss Convergence
    plt.subplot(1, 2, 1)
    plt.plot(optimizer.loss_history_)
    plt.title("Loss Convergence (Log Loss)", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Plot 2: Decision Boundary
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', label='Data Points')
    
    # Create a grid to plot the decision boundary
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    # The decision boundary is where X @ w = 0 (since sigmoid(0) = 0.5)
    # w0 + w1*x1 + w2*x2 = 0  =>  x2 = (-w0 - w1*x1) / w2
    y_vals = -(optimizer.w_[0] + optimizer.w_[1] * x_vals) / optimizer.w_[2]
    
    plt.plot(x_vals, y_vals, color='green', lw=3, label='Decision Boundary')
    plt.title("Logistic Regression Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Done.")