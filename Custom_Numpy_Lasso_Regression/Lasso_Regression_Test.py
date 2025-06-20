import numpy as np
from Lasso_Regression import LassoRegression

# Generate synthetic data
np.random.seed(0)  # For reproducibility
X = np.random.rand(100, 3)  # 100 samples, 3 features
true_coef = np.array([1.5, -2.0, 0.0])  # True coefficients
y = X @ true_coef + np.random.normal(0, 0.1, 100)  # Add some noise


# Create an instance of LassoRegression
lasso = LassoRegression(alpha=0.1)

# Fit the model
lasso.fit(X, y)

# Make predictions
predictions = lasso.predict(X)
from sklearn.metrics import mean_squared_error, r2_score

# Calculate MSE and R-squared
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

print('Learned Coefficients:', lasso.coef_)
print('True Coefficients:', true_coef)