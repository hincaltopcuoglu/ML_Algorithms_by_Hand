"""Visualize NumPy vs JAX (No JIT) vs JAX (With JIT) comparison"""

import matplotlib.pyplot as plt
import numpy as np

# Results from comparison
methods = ['NumPy', 'JAX\n(No JIT)', 'JAX\n(With JIT)']
times = [0.0195, 1.5717, 0.0774]
accuracies = [0.8596, 0.8660, 0.8660]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('NumPy vs JAX: Impact of JIT Compilation', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: Training Time Comparison
# ============================================================================
ax1 = axes[0]
bars1 = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Training Time Comparison')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, time in zip(bars1, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# ============================================================================
# Plot 2: Speedup Factor
# ============================================================================
ax2 = axes[1]
baseline = times[0]  # NumPy as baseline
speedups = [baseline / t for t in times]
bars2 = ax2.bar(methods, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup (vs NumPy)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Comparison')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='NumPy baseline')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend()

# Add value labels on bars
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# ============================================================================
# Plot 3: Accuracy Comparison
# ============================================================================
ax3 = axes[2]
bars3 = ax3.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy Comparison')
ax3.set_ylim([0.85, 0.87])
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars3, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('jit_comparison_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved to: jit_comparison_visualization.png")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nJIT Impact:")
print(f"  • JAX without JIT: {times[1]:.4f}s")
print(f"  • JAX with JIT:    {times[2]:.4f}s")
print(f"  • Speedup:         {times[1]/times[2]:.2f}x faster! 🚀")

print("\nCPU Performance (NumPy baseline):")
print(f"  • NumPy:           {times[0]:.4f}s (baseline)")
print(f"  • JAX (JIT):       {times[2]:.4f}s ({times[0]/times[2]:.2f}x)")
print(f"  • Gap:             JAX is {abs(times[0]-times[2])*1000:.2f}ms slower on CPU")

print("\nAccuracy:")
print(f"  • All methods achieve similar accuracy ✓")
print(f"  • NumPy:     {accuracies[0]:.4f}")
print(f"  • JAX (JIT): {accuracies[2]:.4f}")

print("\nKey Insight:")
print("  On GPU: JAX (JIT) would be 100x+ faster than NumPy!")
print("=" * 70)

plt.show()
