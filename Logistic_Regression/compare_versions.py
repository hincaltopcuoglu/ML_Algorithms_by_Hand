"""Compare NumPy vs JAX logistic regression implementations"""

import sys
import os
sys.path.insert(0, '/Users/hincaltopcuoglu/Desktop/RNN_Coding/Logistic_Regression')

import time
import numpy as np
from sklearn.datasets import make_classification

# Generate same data for both
X, y = make_classification(n_samples=200, n_features=10, random_state=42)

print("=" * 60)
print("LOGISTIC REGRESSION: NumPy vs JAX Comparison")
print("=" * 60)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")

# ============================================================================
# 1. NumPy Version
# ============================================================================
print("1. Running NumPy version...")
start_numpy = time.time()

from logistic_regression import LogisticRegression

lr_numpy = LogisticRegression(learning_rate=0.001, n_iterations=100)
lr_numpy.fit(X, y)
y_pred_numpy = lr_numpy.predict(X)
accuracy_numpy = np.mean(y_pred_numpy == y)
final_loss_numpy = lr_numpy.cost_history[-1]

time_numpy = time.time() - start_numpy

print(f"   ✓ NumPy done in {time_numpy:.4f}s")
print(f"   Final Loss: {final_loss_numpy:.4f}")
print(f"   Accuracy:  {accuracy_numpy:.4f}")

# ============================================================================
# 2. JAX Version
# ============================================================================
print("\n2. Running JAX version...")
start_jax = time.time()

import jax.numpy as jnp
from jax import vmap, grad, random

# Copy functions from JAX implementation
def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))

def predict_single(x, w, b):
    z = jnp.dot(w, x) + b
    return sigmoid(z)

def binary_cross_entropy(y_pred, y_true):
    eps = 1e-7
    return jnp.mean(-(y_true * jnp.log(y_pred + eps) + (1-y_true)*jnp.log(1-y_pred+eps)))

def predict_batch(X, w, b):
    return vmap(lambda x: predict_single(x, w, b))(X)

def loss_fn(w, b, X, y):
    y_pred = predict_batch(X, w, b)
    return binary_cross_entropy(y_pred, y)

def train_step(w, b, X, y, learning_rate):
    loss_value = loss_fn(w, b, X, y)
    grad_w, grad_b = grad(loss_fn, argnums=(0,1))(w, b, X, y)
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new, loss_value

def train_jax(X, y, epochs=100, learning_rate=0.1):
    n_features = X.shape[1]
    key = random.PRNGKey(0)
    w = random.normal(key, (n_features,)) * 0.01
    b = 0.0
    losses = []
    for epoch in range(epochs):
        w, b, loss = train_step(w, b, X, y, learning_rate)
        losses.append(float(loss))
    return w, b, losses

# Convert to JAX arrays
X_jax = jnp.array(X, dtype=jnp.float32)
y_jax = jnp.array(y, dtype=jnp.float32)

# Train JAX model with learning_rate=0.1 (to match JAX default, but 100 iterations)
w_jax, b_jax, losses_jax = train_jax(X_jax, y_jax, epochs=100, learning_rate=0.1)

y_pred_jax = predict_batch(X_jax, w_jax, b_jax)
y_pred_jax = (y_pred_jax > 0.5).astype(int)
accuracy_jax = jnp.mean(y_pred_jax == y_jax)
final_loss_jax = losses_jax[-1]

time_jax = time.time() - start_jax

print(f"   ✓ JAX done in {time_jax:.4f}s")
print(f"   Final Loss: {final_loss_jax:.4f}")
print(f"   Accuracy:  {accuracy_jax:.4f}")

# ============================================================================
# 3. Comparison
# ============================================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'NumPy':<15} {'JAX':<15} {'Difference'}")
print("-" * 60)
print(f"{'Training Time (s)':<20} {time_numpy:.6f}{'':<8} {time_jax:.6f}{'':<8} {time_numpy/time_jax:.2f}x")
print(f"{'Final Loss':<20} {final_loss_numpy:.6f}{'':<8} {final_loss_jax:.6f}{'':<8} {abs(final_loss_numpy - final_loss_jax):.6f}")
print(f"{'Accuracy':<20} {accuracy_numpy:.6f}{'':<8} {float(accuracy_jax):.6f}{'':<8} {abs(accuracy_numpy - float(accuracy_jax)):.6f}")

print("\n" + "=" * 60)
if time_numpy > time_jax:
    print(f"✓ JAX is {time_numpy/time_jax:.2f}x faster")
else:
    print(f"✓ NumPy is {time_jax/time_numpy:.2f}x faster")

if abs(final_loss_numpy - final_loss_jax) < 0.01:
    print("✓ Results are very similar (loss difference < 0.01)")
else:
    print(f"⚠ Results differ (loss difference: {abs(final_loss_numpy - final_loss_jax):.4f})")

print("=" * 60)
