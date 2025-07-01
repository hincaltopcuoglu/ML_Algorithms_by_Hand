from knn_regression import KNNRegressor
import numpy as np

# ======================================================
#                   TESTING THE MODEL
# ======================================================

if __name__ == '__main__':
    # 1. Create some simple sample data
    # We have two groups of points: (1,2,3) with values around 100
    # and (10,11,12) with values around 200.
    X_train = np.array([
        [1], [2], [3], 
        [10], [11], [12]
    ])
    y_train = np.array([100, 105, 110, 200, 205, 210])

    # 2. Instantiate and train the model
    # We'll use k=3, so it should look at the 3 nearest neighbors.
    knn = KNNRegressor(k=3)
    knn.fit(X_train, y_train)

    # 3. Create new data points to predict
    # Let's test a point close to the first group ([4])
    # and a point close to the second group ([9]).
    X_new = np.array([[4], [9]])

    # 4. Make predictions
    predictions = knn.predict(X_new)
    
    # 5. Print the results
    print(f"Training Data X:\n{X_train.flatten()}")
    print(f"Training Data y:\n{y_train}")
    print("-" * 30)
    print(f"Data to Predict: {X_new.flatten()}")
    print(f"Predicted values: {predictions}")
    print("-" * 30)

    # Let's manually verify the first prediction for X_new = [4]
    # Distances from 4: |4-1|=3, |4-2|=2, |4-3|=1, |4-10|=6, |4-11|=7, |4-12|=8
    # The 3 nearest neighbors are the points 3, 2, and 1.
    # Their y-values are 110, 105, and 100.
    # The mean is (110 + 105 + 100) / 3 = 105.0
    # The model should predict 105.
    print(f"Expected prediction for [4]: 105.0")
    print(f"Model prediction for [4]: {predictions[0]}")

    # Let's manually verify the second prediction for X_new = [9]
    # Distances from 9: |9-1|=8, |9-2|=7, |9-3|=6, |9-10|=1, |9-11|=2, |9-12|=3
    # The 3 nearest neighbors are the points 10, 11, and 12.
    # Their y-values are 200, 205, and 210.
    # The mean is (200 + 205 + 210) / 3 = 205.0
    # The model should predict 205.
    print(f"Expected prediction for [9]: 205.0")
    print(f"Model prediction for [9]: {predictions[1]}")