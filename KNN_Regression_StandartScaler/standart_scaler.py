import numpy as np

class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None # this is the standart deviation

    def fit(self,X):
        self.mean_ = np.mean(X,axis=0)
        self.scale_ = np.std(X,axis=0)

    def transform(self,X):
        return (X - self.mean_) / self.scale_

    
    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)