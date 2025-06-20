import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error, r2_score
from Lasso_Regression import LassoRegression

# Load data from CSV
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding='utf-8', names=True)
    return data

# Handle NaN values in a NumPy way
def handle_nans(data):
    # Create a mask for valid rows (no NaNs)
    valid_mask = np.ones(len(data),dtype=bool)
    for name in data.dtype.names:
        if np.issubdtype(data[name].dtype, np.number):
            valid_mask &= ~np.isnan(data[name])
    return data[valid_mask]



def one_hot_encode(data):
    # Extract categorical columns
    sex = data['sex']
    smoker = data['smoker']
    region = data['region']

    # One-hot encode 'sex'
    sex_unique = np.unique(sex)
    sex_encoded = np.zeros((len(sex), len(sex_unique)))
    for i, category in enumerate(sex):
        sex_encoded[i, np.where(sex_unique == category)[0][0]] = 1

    # One-hot encode 'smoker'
    smoker_unique = np.unique(smoker)
    smoker_encoded = np.zeros((len(smoker), len(smoker_unique)))
    for i, category in enumerate(smoker):
        smoker_encoded[i, np.where(smoker_unique == category)[0][0]] = 1

    # One-hot encode 'region'
    region_unique = np.unique(region)
    region_encoded = np.zeros((len(region), len(region_unique)))
    for i, category in enumerate(region):
        region_encoded[i, np.where(region_unique == category)[0][0]] = 1

    # Extract numerical features and convert to float
    age = data['age'].astype(float)
    bmi = data['bmi'].astype(float)
    children = data['children'].astype(float)

    # Combine numerical features
    numerical_features = np.column_stack((age, bmi, children))
    X = np.hstack((numerical_features, sex_encoded, smoker_encoded, region_encoded))
    return X

# Load your data
data = load_data('insurance.csv')  # Replace with your actual CSV file path

# Handle NaN values
data = handle_nans(data)


# Prepare the data
X = one_hot_encode(data)
y = data['expenses'].astype(float)  # Use the actual column name for expenses


print("Any NaNs in X?", np.isnan(X).any())
print("Any NaNs in y?", np.isnan(y).any())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Timing and evaluating scikit-learn Lasso
start_time = time.time()
sklearn_lasso = SklearnLasso(alpha=1.0)
sklearn_lasso.fit(X_train, y_train)
y_pred_sklearn = sklearn_lasso.predict(X_test)
sklearn_time = time.time() - start_time

# Timing and evaluating custom Lasso
start_time = time.time()
custom_lasso = LassoRegression(alpha=1.0)
custom_lasso.fit(X_train, y_train)
y_pred_custom = custom_lasso.predict(X_test)
custom_time = time.time() - start_time

# Calculate metrics
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

mse_custom = mean_squared_error(y_test, y_pred_custom)
r2_custom = r2_score(y_test, y_pred_custom)

# Print results
print(f"Scikit-learn Lasso: Time = {sklearn_time:.4f}s, MSE = {mse_sklearn:.4f}, R² = {r2_sklearn:.4f}")
print(f"Custom Lasso: Time = {custom_time:.4f}s, MSE = {mse_custom:.4f}, R² = {r2_custom:.4f}")