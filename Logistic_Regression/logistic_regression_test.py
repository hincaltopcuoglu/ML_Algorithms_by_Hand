import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegression

# Example of how to use it
if __name__ == '__main__':

    # 1. Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # 3. Create and train your model
    model = LogisticRegression(learning_rate=0.0001, n_iterations=1000)
    model.fit(X_train, y_train)

    # 4. Make predictions
    predictions = model.predict(X_test)

    # 5. Check accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Logistic Regression accuracy: {accuracy:.4f}")

    # You could also plot model.cost_history here to see the learning curve!