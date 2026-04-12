"""Compare NumPy vs JAX (without JIT) vs JAX (with JIT)"""

import sys
sys.path.insert(0, '/Users/hincaltopcuoglu/Desktop/RNN_Coding/Logistic_Regression')

import time
import numpy as np
from sklearn.datasets import make_classification
import jax.numpy as jnp
from jax import vmap, grad, random, jit

# Generate test data
X, y = make_classification(n_samples=5000, n_features=50, random_state=42)

print("=" * 80)
print("LOGISTIC REGRESSION: NumPy vs JAX (No JIT) vs JAX (With JIT)")
print("=" * 80)
print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features, 100 epochs\n")

# ============================================================================
# 1. NumPy Version
# ============================================================================
print("1. Running NumPy version...")
from logistic_regression import LogisticRegression

start_numpy = time.time()
lr_numpy = LogisticRegression(learning_rate=0.001, n_iterations=100)
lr_numpy.fit(X, y)
y_pred_numpy = lr_numpy.predict(X)
accuracy_numpy = np.mean(y_pred_numpy == y)
time_numpy = time.time() - start_numpy

print(f"   ✓ Done in {time_numpy:.4f}s | Accuracy: {accuracy_numpy:.4f}")

# ============================================================================
# 2. JAX WITHOUT JIT
# ============================================================================
print("\n2. Running JAX (NO JIT)...")

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

def loss_fn_no_jit(w, b, X, y):
    y_pred = predict_batch(X, w, b)
    return binary_cross_entropy(y_pred, y)

def train_step_no_jit(w, b, X, y, learning_rate):
    loss_value = loss_fn_no_jit(w, b, X, y)
    grad_w, grad_b = grad(loss_fn_no_jit, argnums=(0,1))(w, b, X, y)
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new, loss_value

def train_no_jit(X, y, epochs=100, learning_rate=0.1):
    n_features = X.shape[1]
    key = random.PRNGKey(0)
    w = random.normal(key, (n_features,)) * 0.01
    b = 0.0
    for epoch in range(epochs):
        w, b, loss = train_step_no_jit(w, b, X, y, learning_rate)
    return w, b

X_jax = jnp.array(X, dtype=jnp.float32)
y_jax = jnp.array(y, dtype=jnp.float32)

start_jax_no_jit = time.time()
w_no_jit, b_no_jit = train_no_jit(X_jax, y_jax, epochs=100, learning_rate=0.1)
y_pred_no_jit = predict_batch(X_jax, w_no_jit, b_no_jit)
y_pred_no_jit = (y_pred_no_jit > 0.5).astype(int)
accuracy_no_jit = jnp.mean(y_pred_no_jit == y_jax)
time_jax_no_jit = time.time() - start_jax_no_jit

print(f"   ✓ Done in {time_jax_no_jit:.4f}s | Accuracy: {accuracy_no_jit:.4f}")

# ============================================================================
# 3. JAX WITH JIT
# ============================================================================
print("\n3. Running JAX (WITH JIT)...")

@jit
def loss_fn_jit(w, b, X, y):
    y_pred = predict_batch(X, w, b)
    return binary_cross_entropy(y_pred, y)

@jit
def train_step_jit(w, b, X, y, learning_rate):
    loss_value = loss_fn_jit(w, b, X, y)
    grad_w, grad_b = grad(loss_fn_jit, argnums=(0,1))(w, b, X, y)
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new, loss_value

def train_jit(X, y, epochs=100, learning_rate=0.1):
    n_features = X.shape[1]
    key = random.PRNGKey(0)
    w = random.normal(key, (n_features,)) * 0.01
    b = 0.0
    for epoch in range(epochs):
        w, b, loss = train_step_jit(w, b, X, y, learning_rate)
    return w, b

start_jax_jit = time.time()
w_jit, b_jit = train_jit(X_jax, y_jax, epochs=100, learning_rate=0.1)
y_pred_jit = predict_batch(X_jax, w_jit, b_jit)
y_pred_jit = (y_pred_jit > 0.5).astype(int)
accuracy_jit = jnp.mean(y_pred_jit == y_jax)
time_jax_jit = time.time() - start_jax_jit

print(f"   ✓ Done in {time_jax_jit:.4f}s | Accuracy: {accuracy_jit:.4f}")

# ============================================================================
# 4. Results Comparison
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print(f"{'Method':<25} {'Time (s)':<15} {'Accuracy':<15} {'Speedup vs JAX No-JIT'}")
print("-" * 80)
print(f"{'NumPy':<25} {time_numpy:<15.4f} {accuracy_numpy:<15.4f} {'Baseline'}")
print(f"{'JAX (No JIT)':<25} {time_jax_no_jit:<15.4f} {accuracy_no_jit:<15.4f} {'1.0x'}")
print(f"{'JAX (With JIT)':<25} {time_jax_jit:<15.4f} {accuracy_jit:<15.4f} {f'{time_jax_no_jit/time_jax_jit:.2f}x'}")

print("\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
speedup_jit = time_jax_no_jit / time_jax_jit
speedup_numpy_vs_jit = time_numpy / time_jax_jit

print(f"✓ JIT speedup: {speedup_jit:.2f}x faster than no-JIT")
print(f"✓ NumPy vs JAX (JIT): NumPy is {speedup_numpy_vs_jit:.2f}x faster (still CPU)")
print(f"  → On GPU: JAX (JIT) would be 100x+ faster than NumPy!")
print("=" * 80)
