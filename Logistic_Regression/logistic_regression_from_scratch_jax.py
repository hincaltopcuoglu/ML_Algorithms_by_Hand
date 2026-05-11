import jax.numpy as jnp
from jax import vmap
from jax import grad
from jax import random
from jax import jit

def sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def predict_single(x, w, b):
    z = jnp.dot(w, x) + b
    return sigmoid(z)


def binary_cross_entropy(y_pred, y_true):
    eps = 1e-7
    return jnp.mean(-(y_true * jnp.log(y_pred + eps) + (1- y_true)*jnp.log(1- y_pred +eps)))


def predict_batch(X, w, b):
    return vmap(lambda x: predict_single(x, w, b))(X)

@jit
def loss_fn(w, b, X, y):
    y_pred = predict_batch(X, w, b)
    return binary_cross_entropy(y_pred, y)

@jit
def train_step(w, b, X, y, learning_rate):
    loss_value = loss_fn(w, b, X, y)
    grad_w, grad_b = grad(loss_fn, argnums=(0,1))(w, b, X, y)
    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b
    return w_new, b_new, loss_value


def train(X, y, epochs=100, learning_rate=0.1):
    n_features = X.shape[1]

    # initialize parameters
    key = random.PRNGKey(0)
    w = random.normal(key, (n_features,)) * 0.01 # small random values
    b = 0.0

    losses = []

    for epoch in range(epochs):
        w, b, loss = train_step(w, b, X, y, learning_rate)
        losses.append(float(loss))

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.4f}")

    return w, b, losses

if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=10, random_state=42)

    # Convert to Jax arrays
    X = jnp.array(X, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)

    w, b, losses = train(X, y, epochs=100, learning_rate=0.1)

    # Make predictions
    y_pred_probs = predict_batch(X, w, b)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Accuracy
    accuracy = jnp.mean(y_pred == y)
    print(f"\nFinal accuracy: {accuracy:.4f}")