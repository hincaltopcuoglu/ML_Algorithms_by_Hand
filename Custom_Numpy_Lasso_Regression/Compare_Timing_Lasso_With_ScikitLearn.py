import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error, r2_score
import time
from Lasso_Regression import LassoRegression

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
true_coef = np.array([1.5, -2.0, 0.0])
y = X @ true_coef + np.random.normal(0, 0.1, 100)  # Add some noise

# Timing your Lasso implementation
lasso = LassoRegression(alpha=0.1)

start_time = time.time()
lasso.fit(X, y)
end_time = time.time()

# Make predictions
predictions = lasso.predict(X)

# Calculate metrics
mse_custom = mean_squared_error(y, predictions)
r2_custom = r2_score(y, predictions)

print(f'Custom Lasso Mean Squared Error: {mse_custom}')
print(f'Custom Lasso R-squared: {r2_custom}')
print(f'Custom Lasso Time: {end_time - start_time:.4f} seconds')

# Timing scikit-learn Lasso implementation
sklearn_lasso = SklearnLasso(alpha=0.1)

start_time = time.time()
sklearn_lasso.fit(X, y)
end_time = time.time()

# Make predictions
sklearn_predictions = sklearn_lasso.predict(X)

# Calculate metrics
mse_sklearn = mean_squared_error(y, sklearn_predictions)
r2_sklearn = r2_score(y, sklearn_predictions)

print(f'Scikit-learn Lasso Mean Squared Error: {mse_sklearn}')
print(f'Scikit-learn Lasso R-squared: {r2_sklearn}')
print(f'Scikit-learn Lasso Time: {end_time - start_time:.4f} seconds')


print('Custom Lasso Coefficients:', lasso.coef_)
print('Scikit-learn Lasso Coefficients:', sklearn_lasso.coef_)