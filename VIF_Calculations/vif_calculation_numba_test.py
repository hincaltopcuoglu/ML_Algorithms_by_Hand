import numpy as np
import pandas as pd
import time
from vif_calculation_numba import compute_vif_numba

# Simulate test data: 2M rows × 30 cols
np.random.seed(42)
n_rows = 2_000_000
n_cols = 30

print("Generating synthetic data...")
X_test = np.random.randn(n_rows, n_cols).astype(np.float32)

# Center columns
X_test -= X_test.mean(axis=0)

# Time VIF calculation
print("Computing VIF using Numba...")
start = time.time()
vif_values = compute_vif_numba(X_test)
end = time.time()

# Show results
col_names = [f"x{i}" for i in range(n_cols)]
vif_series = pd.Series(vif_values, index=col_names, name="VIF")

print(f"\nTime taken: {end - start:.2f} seconds")
print("\nTop 10 VIF values:")
print(vif_series.sort_values(ascending=False).head(10))
