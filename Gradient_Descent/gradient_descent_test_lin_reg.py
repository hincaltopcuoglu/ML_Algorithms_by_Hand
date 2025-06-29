import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import GradientDescent

# ===================================================================
# PROBLEM-SPECIFIC FUNCTIONS (For Linear Regression)
# ===================================================================

def mse_loss(X, y, w):
    """Mean Squared Error loss for a linear model."""
    m = len(y)
    y_pred = X @ w
    error = y_pred - y
    return (1 / m) * np.sum(error**2)

def mse_gradient(X, y, w):
    """Gradient of the MSE loss."""
    m = len(y)
    y_pred = X @ w
    error = y_pred - y
    return (2 / m) * X.T @ error

# ===================================================================
# MAIN EXECUTION BLOCK (The Test)
# ===================================================================
if __name__ == "__main__":
    # --- Step 1: Generate synthetic data ---
    print("--- 1. Generating Data ---")
    np.random.seed(42)  # for reproducible results
    n_samples = 100
    
    # Create X with a bias term (a column of ones for the intercept)
    X_raw = 2 * np.random.rand(n_samples, 1)
    X_b = np.c_[np.ones((n_samples, 1)), X_raw]  # Add x0 = 1 to each instance
    
    # Define the "true" weights we want the algorithm to find
    # w0 (bias/intercept) = 4.0, w1 (slope) = 3.0
    true_w = np.array([4.0, 3.0]) 
    
    # Create y using the true weights plus some random noise
    y = X_b @ true_w + np.random.randn(n_samples)

    # --- Step 2: Initialize and run the optimizer ---
    print("\n--- 2. Running Optimizer ---")
    optimizer = GradientDescent(loss_fn=mse_loss, grad_fn=mse_gradient, lr=0.1, max_iter=1000)
    
    # Start with a random guess for the weights
    initial_w = np.random.randn(X_b.shape[1])
    
    # Run the optimization!
    optimizer.optimize(X_b, y, initial_w)
    
    # --- Step 3: Inspect and visualize the results ---
    print("\n--- 3. Results ---")
    print(f"True weights:      {true_w}")
    print(f"Optimized weights: {optimizer.w_}")

    # --- Step 4: Visualize the convergence and the final fit ---
    print("\n--- 4. Generating Plots ---")
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Loss Convergence
    plt.subplot(1, 2, 1)
    plt.plot(optimizer.loss_history_)
    plt.title("Loss Convergence", fontsize=14)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.grid(True)
    
    # Plot 2: Linear Regression Fit
    plt.subplot(1, 2, 2)
    plt.scatter(X_raw, y, alpha=0.7, label="Original Data")
    y_pred_final = X_b @ optimizer.w_
    plt.plot(X_raw, y_pred_final, color='red', linewidth=3, label="Fitted Line (Our GD)")
    plt.title("Linear Regression Fit", fontsize=14)
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (y)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print("Done.")