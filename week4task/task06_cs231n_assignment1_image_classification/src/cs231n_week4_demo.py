import numpy as np


def make_tiny_image_dataset(samples_per_class=80, image_size=8, noise=0.28, seed=7):
    """Create a small image-like dataset with three simple visual classes."""
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    for label in range(3):
        for _ in range(samples_per_class):
            img = rng.normal(0.0, noise, size=(image_size, image_size))
            if label == 0:
                img[:, image_size // 2 - 1 : image_size // 2 + 1] += 1.0
            elif label == 1:
                img[image_size // 2 - 1 : image_size // 2 + 1, :] += 1.0
            else:
                np.fill_diagonal(img, img.diagonal() + 1.0)
                np.fill_diagonal(np.fliplr(img), np.fliplr(img).diagonal() + 0.7)
            images.append(img)
            labels.append(label)

    X = np.asarray(images).reshape(-1, image_size * image_size)
    y = np.asarray(labels)
    order = rng.permutation(len(y))
    return X[order], y[order]


def split_data(X, y, train_ratio=0.65, val_ratio=0.15):
    n_train = int(len(y) * train_ratio)
    n_val = int(len(y) * val_ratio)
    return (
        X[:n_train],
        y[:n_train],
        X[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
        X[n_train + n_val :],
        y[n_train + n_val :],
    )


def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


class KNearestNeighbor:
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=3):
        test_sq = np.sum(X * X, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train * self.X_train, axis=1)
        dists = np.sqrt(np.maximum(test_sq + train_sq - 2 * X @ self.X_train.T, 0.0))
        nearest = np.argsort(dists, axis=1)[:, :k]
        votes = self.y_train[nearest]
        return np.array([np.bincount(row, minlength=3).argmax() for row in votes])


def softmax_loss_and_grad(W, X, y, reg):
    scores = X @ W
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)

    n = X.shape[0]
    loss = -np.log(probs[np.arange(n), y]).mean() + reg * np.sum(W * W)

    dscores = probs
    dscores[np.arange(n), y] -= 1
    dscores /= n
    grad = X.T @ dscores + 2 * reg * W
    return loss, grad


def train_softmax(X_train, y_train, X_val, y_val, num_classes, lr=0.2, reg=1e-3, epochs=700):
    rng = np.random.default_rng(11)
    W = rng.normal(0, 0.01, size=(X_train.shape[1], num_classes))
    history = []

    for epoch in range(epochs):
        loss, grad = softmax_loss_and_grad(W, X_train, y_train, reg)
        W -= lr * grad
        if epoch % 100 == 0 or epoch == epochs - 1:
            val_acc = accuracy(predict_linear(W, X_val), y_val)
            history.append((epoch, loss, val_acc))
    return W, history


def predict_linear(W, X):
    return np.argmax(X @ W, axis=1)


def train_two_layer_net(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dim=32,
    lr=0.08,
    reg=1e-3,
    epochs=900,
    seed=23,
):
    rng = np.random.default_rng(seed)
    num_classes = int(y_train.max()) + 1
    W1 = rng.normal(0, 0.05, size=(X_train.shape[1], hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, 0.05, size=(hidden_dim, num_classes))
    b2 = np.zeros(num_classes)
    history = []

    for epoch in range(epochs):
        hidden = np.maximum(0, X_train @ W1 + b1)
        scores = hidden @ W2 + b2
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)

        n = X_train.shape[0]
        loss = -np.log(probs[np.arange(n), y_train]).mean()
        loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        dscores = probs
        dscores[np.arange(n), y_train] -= 1
        dscores /= n
        dW2 = hidden.T @ dscores + reg * W2
        db2 = dscores.sum(axis=0)
        dhidden = dscores @ W2.T
        dhidden[hidden <= 0] = 0
        dW1 = X_train.T @ dhidden + reg * W1
        db1 = dhidden.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 150 == 0 or epoch == epochs - 1:
            val_pred = predict_two_layer(X_val, W1, b1, W2, b2)
            history.append((epoch, loss, accuracy(val_pred, y_val)))

    return (W1, b1, W2, b2), history


def predict_two_layer(X, W1, b1, W2, b2):
    hidden = np.maximum(0, X @ W1 + b1)
    return np.argmax(hidden @ W2 + b2, axis=1)


def accuracy(pred, y):
    return float(np.mean(pred == y))


def main():
    X, y = make_tiny_image_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)

    print("CS231n Assignment 1 mini demo")
    print(f"train/val/test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")

    knn = KNearestNeighbor()
    knn.train(X_train, y_train)
    for k in (1, 3, 5):
        print(f"kNN k={k}: val_acc={accuracy(knn.predict(X_val, k=k), y_val):.3f}")

    W_softmax, softmax_history = train_softmax(X_train, y_train, X_val, y_val, 3)
    print("Softmax checkpoints:")
    for epoch, loss, val_acc in softmax_history:
        print(f"  epoch={epoch:03d} loss={loss:.4f} val_acc={val_acc:.3f}")
    print(f"Softmax test_acc={accuracy(predict_linear(W_softmax, X_test), y_test):.3f}")

    params, nn_history = train_two_layer_net(X_train, y_train, X_val, y_val)
    print("Two-layer net checkpoints:")
    for epoch, loss, val_acc in nn_history:
        print(f"  epoch={epoch:03d} loss={loss:.4f} val_acc={val_acc:.3f}")
    print(f"Two-layer net test_acc={accuracy(predict_two_layer(X_test, *params), y_test):.3f}")


if __name__ == "__main__":
    main()
