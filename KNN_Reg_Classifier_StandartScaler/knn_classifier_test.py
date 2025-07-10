from sklearn.datasets import make_blobs # To generate sample data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

from knn_classifier import KNeighboursClassifier
from standart_scaler import StandardScaler

if __name__ == "__main__":
    print("Testing custom KNeighborsClassifier...")


    # 1. Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target


    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Instantiate and train our classifier
    k_value = 5
    
    # --- Test with 'uniform' weights (should be the same as before) ---
    print(f"--- Testing with k={k_value} and weights='uniform' ---")
    classifier_uniform = KNeighboursClassifier(k=k_value, weights='uniform')
    classifier_uniform.fit(X_train, y_train)
    predictions_uniform = classifier_uniform.predict(X_test)
    accuracy_uniform = accuracy_score(y_test, predictions_uniform)
    print(f"Accuracy with uniform weights: {accuracy_uniform:.4f}\n")

    # --- Test with 'distance' weights ---
    print(f"--- Testing with k={k_value} and weights='distance' ---")
    classifier_distance = KNeighboursClassifier(k=k_value, weights='distance')
    classifier_distance.fit(X_train, y_train)
    predictions_distance = classifier_distance.predict(X_test)
    accuracy_distance = accuracy_score(y_test, predictions_distance)
    print(f"Accuracy with distance weights: {accuracy_distance:.4f}")

    # For comparison, let's see what scikit-learn gets for both
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    
    sk_uniform = SklearnKNN(n_neighbors=k_value, weights='uniform').fit(X_train, y_train)
    print(f"Scikit-learn (uniform) accuracy: {sk_uniform.score(X_test, y_test):.4f}")

    sk_distance = SklearnKNN(n_neighbors=k_value, weights='distance').fit(X_train, y_train)
    print(f"Scikit-learn (distance) accuracy: {sk_distance.score(X_test, y_test):.4f}")


