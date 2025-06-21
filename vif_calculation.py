import numpy as np

def compute_vif_numpy(X):
    """
    Compute VIF for each feature using only NumPy.
    
    Parameters:
        X: ndarray of shape (n_samples, n_features)
           Each column is a feature. Should be centered (zero mean) for stability.
    
    Returns:
        vif: ndarray of shape (n_features,)
             VIF value for each feature.
    """

    n,k = X.shape
    vif = np.zeros(k)

    for i in range(k):
        X_i = X[:, i]
        X_others = np.delete(X_i, axis=1)

        """
        
        In the context of VIF, for each variable X_i,
        we regress it as the response on all the other variables as predictors.
        So the formuka is:
        X_i = X_others * Beta + epsilon

        X_i = the column vector to be predicted (think it as y)
        X_others = all the other columns of the matrix X, used as features
        Beta = coefficient of linear model
        """

        # Solve least squares: X_i ≈ X_others @ beta
        beta, residuals, rank, s = np.linalg.lstsq(X_others,X_i, rcond=None)
        y_pred = X_others @ beta
        ss_res = np.sum((X_i - y_pred)**2)
        ss_tot = np.sum((X_i - np.mean(X_i))**2)

        r2 = 1 - ss_res / ss_tot
        vif[i] = 1 / (1 - r2) if r2 < 1 else np.inf  # Avoid division by zero
    
    return vif
