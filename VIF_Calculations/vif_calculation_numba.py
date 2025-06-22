import numpy as np
import pandas as pd
from numba import njit
import time

@njit
def compute_vif_numba(X):
    n, k = X.shape
    vif = np.zeros(k, dtype=np.float32)

    for i in range(k):
        X_i = X[:, i]
        X_others = np.empty((n, k - 1), dtype=np.float32)

        col = 0
        for j in range(k):
            if j != i:
                X_others[:, col] = X[:, j]
                col += 1

        # QR decomposition (faster & stable)
        Q, R = np.linalg.qr(X_others)
        beta = np.linalg.solve(R, Q.T @ X_i)
        y_pred = X_others @ beta

        ss_res = np.sum((X_i - y_pred) ** 2)
        ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)

        r2 = 1 - ss_res / ss_tot
        vif[i] = 1 / (1 - r2) if r2 < 1 else np.inf

    return vif
