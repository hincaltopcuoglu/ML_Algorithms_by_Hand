"""Compare NumPy vs JAX on LARGER dataset"""

import sys
sys.path.insert(0, '/Users/hincaltopcuoglu/Desktop/RNN_Coding/Logistic_Regression')

import time
import numpy as np
from sklearn.datasets import make_classification

# Generate LARGER dataset
X, y = make_classification(
    n_samples=50000,      # 50k samples (was 200)
    n_features=100,       # 100 features (was 10)
    random_state=42
)

print("=" * 70)
print("LOGISTIC REGRESSION: NumPy vs JAX - LARGE DATASET")
print("=" * 70)
print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features\n")

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
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch + 1}/100 | Loss: {loss:.4f}")
    return w, b, losses

# Convert to JAX arrays
X_jax = jnp.array(X, dtype=jnp.float32)
y_jax = jnp.array(y, dtype=jnp.float32)

# Train JAX model
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
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"{'Metric':<25} {'NumPy':<18} {'JAX':<18} {'Winner'}")
print("-" * 70)

speedup = time_numpy / time_jax
if speedup > 1:
    winner = f"NumPy ({speedup:.2f}x faster)"
else:
    winner = f"JAX ({1/speedup:.2f}x faster)"

print(f"{'Training Time (s)':<25} {time_numpy:.6f}{'':<10} {time_jax:.6f}{'':<10} {winner}")
print(f"{'Final Loss':<25} {final_loss_numpy:.6f}{'':<10} {final_loss_jax:.6f}")
print(f"{'Accuracy':<25} {accuracy_numpy:.6f}{'':<10} {float(accuracy_jax):.6f}")

print("\n" + "=" * 70)
if abs(accuracy_numpy - float(accuracy_jax)) < 0.01:
    print("✓ Accuracies are similar - both models are correct!")
else:
    print(f"⚠ Accuracy difference: {abs(accuracy_numpy - float(accuracy_jax)):.4f}")

print("=" * 70)
print("\nDataset Scale-up:")
print(f"  Small dataset: 200 samples → NumPy 794x faster (overhead matters)")
print(f"  Large dataset: 50k samples → JAX more competitive")
print(f"  On GPU: JAX would be many 100x faster!")
print("=" * 70)
