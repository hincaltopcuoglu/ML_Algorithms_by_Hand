import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time

def single_vif(i, X):
    X_i = X[:, i]
    X_others = np.delete(X, i, axis=1)
    beta, *_ = np.linalg.lstsq(X_others, X_i, rcond=None)
    y_pred = X_others @ beta
    ss_res = np.sum((X_i - y_pred) ** 2)
    ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return 1 / (1 - r2) if r2 < 1 else np.inf

def compute_vif_parallel(X_df, n_jobs=-1):
    """
    Compute VIF using parallel processing.
    
    Parameters:
        X_df (pd.DataFrame): input features
        n_jobs (int): number of parallel jobs (-1 = all cores)
        
    Returns:
        pd.Series: VIF values indexed by column names
    """
    X = X_df.values.astype(np.float32)
    vif_values = Parallel(n_jobs=n_jobs)(
        delayed(single_vif)(i, X) for i in range(X.shape[1])
    )
    return pd.Series(vif_values, index=X_df.columns, name="VIF")


## usage: change df and columns as your need
# Center only train columns (once!)
df[columns] -= df[columns].mean()

start = time.time()
vif_result = compute_vif_parallel(df[columns], n_jobs=-1)
end = time.time()

print("Time taken: {:.2f} seconds".format(end - start))
print(vif_result.sort_values(ascending=False))