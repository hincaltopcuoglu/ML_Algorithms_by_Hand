"""Visualize NumPy vs JAX comparisons with plots"""

import sys
sys.path.insert(0, '/Users/hincaltopcuoglu/Desktop/RNN_Coding/Logistic_Regression')

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from logistic_regression import LogisticRegression
import jax.numpy as jnp
from jax import vmap, grad, random

# Helper functions (JAX implementation)
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

# ============================================================================
# Test on different dataset sizes
# ============================================================================

print("Testing on multiple dataset sizes...")
sizes = [200, 500, 1000, 5000, 10000]
results = {
    'size': [],
    'numpy_time': [],
    'jax_time': [],
    'numpy_acc': [],
    'jax_acc': []
}

for size in sizes:
    print(f"\n  Testing with {size} samples...", end=" ")

    X, y = make_classification(n_samples=size, n_features=20, random_state=42)

    # NumPy
    start = time.time()
    lr_numpy = LogisticRegression(learning_rate=0.001, n_iterations=50)
    lr_numpy.fit(X, y)
    y_pred_numpy = lr_numpy.predict(X)
    acc_numpy = np.mean(y_pred_numpy == y)
    time_numpy = time.time() - start

    # JAX
    X_jax = jnp.array(X, dtype=jnp.float32)
    y_jax = jnp.array(y, dtype=jnp.float32)
    start = time.time()
    w_jax, b_jax, _ = train_jax(X_jax, y_jax, epochs=50, learning_rate=0.1)
    y_pred_jax = predict_batch(X_jax, w_jax, b_jax)
    y_pred_jax = (y_pred_jax > 0.5).astype(int)
    acc_jax = jnp.mean(y_pred_jax == y_jax)
    time_jax = time.time() - start

    results['size'].append(size)
    results['numpy_time'].append(time_numpy)
    results['jax_time'].append(time_jax)
    results['numpy_acc'].append(acc_numpy)
    results['jax_acc'].append(float(acc_jax))

    print(f"NumPy: {time_numpy:.4f}s, JAX: {time_jax:.4f}s")

# ============================================================================
# Create visualizations
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NumPy vs JAX Logistic Regression Comparison', fontsize=16, fontweight='bold')

# Plot 1: Training Time vs Dataset Size
ax1 = axes[0, 0]
ax1.plot(results['size'], results['numpy_time'], 'o-', linewidth=2, markersize=8, label='NumPy', color='#1f77b4')
ax1.plot(results['size'], results['jax_time'], 's-', linewidth=2, markersize=8, label='JAX', color='#ff7f0e')
ax1.set_xlabel('Dataset Size (samples)', fontsize=11)
ax1.set_ylabel('Training Time (seconds)', fontsize=11)
ax1.set_title('Training Time vs Dataset Size')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Accuracy Comparison
ax2 = axes[0, 1]
x = np.arange(len(results['size']))
width = 0.35
ax2.bar(x - width/2, results['numpy_acc'], width, label='NumPy', color='#1f77b4', alpha=0.8)
ax2.bar(x + width/2, results['jax_acc'], width, label='JAX', color='#ff7f0e', alpha=0.8)
ax2.set_xlabel('Dataset Size', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.set_title('Accuracy Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(results['size'])
ax2.legend(fontsize=10)
ax2.set_ylim([0.8, 1.0])
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Training Time Speedup (NumPy / JAX)
ax3 = axes[1, 0]
speedup = np.array(results['numpy_time']) / np.array(results['jax_time'])
colors = ['green' if s > 1 else 'red' for s in speedup]
ax3.bar(range(len(results['size'])), speedup, color=colors, alpha=0.7)
ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Break-even')
ax3.set_xlabel('Dataset Size', fontsize=11)
ax3.set_ylabel('Speedup (NumPy / JAX)', fontsize=11)
ax3.set_title('NumPy vs JAX Speedup')
ax3.set_xticks(range(len(results['size'])))
ax3.set_xticklabels(results['size'])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_yscale('log')

# Plot 4: Training Time Ratio
ax4 = axes[1, 1]
ax4.plot(results['size'], results['numpy_time'], 'o-', linewidth=2, markersize=8, label='NumPy', color='#1f77b4')
ax4.plot(results['size'], results['jax_time'], 's-', linewidth=2, markersize=8, label='JAX', color='#ff7f0e')
ax4.set_xlabel('Dataset Size (samples)', fontsize=11)
ax4.set_ylabel('Training Time (seconds)', fontsize=11)
ax4.set_title('Training Time (Log Scale)')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('comparison_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: comparison_visualization.png")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Dataset Size':<15} {'NumPy Time':<15} {'JAX Time':<15} {'Speedup':<15} {'NumPy Acc':<12} {'JAX Acc':<12}")
print("-"*80)
for i, size in enumerate(results['size']):
    speedup = results['numpy_time'][i] / results['jax_time'][i]
    print(f"{size:<15} {results['numpy_time'][i]:<15.4f} {results['jax_time'][i]:<15.4f} {speedup:<15.2f}x {results['numpy_acc'][i]:<12.4f} {results['jax_acc'][i]:<12.4f}")

print("="*80)
plt.show()
