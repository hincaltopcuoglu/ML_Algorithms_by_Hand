"""
The Test Scenario
Training Data: We'll have two types of houses:
Small houses with few rooms (e.g., 2-3 rooms, ~500 sqft) and low prices (~150k).
Large houses with many rooms (e.g., 7-8 rooms, ~2000 sqft) and high prices (~400k).
Test Data: We want to predict the price of a house with 4 rooms and 1000 sqft. 
Intuitively, this house is "in between", but it's closer in spirit to the small houses. Its price should be somewhere above the small houses but well below the large ones.
"""
from knn_regression import KNNRegressor
import numpy as np
from standart_scaler import StandardScaler


# ======================================================
#                   TESTING WITH SCALING
# ======================================================

if __name__ == '__main__':
    # --- 1. The Data ---
    # Feature 1: Rooms (small scale), Feature 2: Area (large scale)
    X_train = np.array([
        [2, 500], [3, 600], [2, 550],  # Small houses
        [7, 2000], [8, 2100], [7, 1900] # Large houses
    ])
    # Target: Price (in thousands)
    y_train = np.array([150, 160, 155, 400, 420, 390])

    # The house we want to predict the price for
    X_new = np.array([[4, 1000]]) # 4 rooms, 1000 sqft

    # We will use k=3
    k = 3
    
    print("="*40)
    print("SCENARIO 1: PREDICTION WITHOUT SCALING")
    print("="*40)

    # --- 2. Predict on Raw, Unscaled Data ---
    knn_raw = KNNRegressor(k=k)
    knn_raw.fit(X_train, y_train)
    prediction_raw = knn_raw.predict(X_new)

    print(f"Test House: {X_new.flatten()}")
    print(f"Prediction on RAW data: ${prediction_raw[0]:.2f}k")
    print("\nExplanation:")
    print("The algorithm is dominated by the 'area' feature. The 3 houses with areas closest to 1000sqft are [500, 600, 550], completely ignoring the 'rooms' feature. So it predicts a price very close to the small houses.")
    # The average of [150, 160, 155] is 155.
    print(f"Expected neighbors are the small houses. Prediction should be ~155k.")


    print("\n" + "="*40)
    print("SCENARIO 2: PREDICTION WITH YOUR StandardScaler")
    print("="*40)

    # --- 3. Scale the Data and Predict Again ---
    # Create and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # IMPORTANT: Use the SAME scaler to transform the new data
    X_new_scaled = scaler.transform(X_new)

    # Print the scaled data to see the difference
    print("Original training data sample:\n", X_train[0])
    print("Scaled training data sample:\n", X_train_scaled[0])
    print("\nOriginal new data:\n", X_new)
    print("Scaled new data:\n", X_new_scaled)

    # Train a new KNN on the SCALED data
    knn_scaled = KNNRegressor(k=k)
    knn_scaled.fit(X_train_scaled, y_train)
    prediction_scaled = knn_scaled.predict(X_new_scaled)
    
    print(f"\nTest House: {X_new.flatten()}")
    print(f"Prediction on SCALED data: ${prediction_scaled[0]:.2f}k")
    print("\nExplanation:")
    print("Now both 'rooms' and 'area' are on a level playing field. The algorithm can correctly identify that [4 rooms, 1000 sqft] is geometrically between the small and large house clusters, resulting in a more balanced and intuitive prediction.")