import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from cs231n.classifiers.linear_classifier import Softmax
from cs231n.layers import (
    affine_backward,
    affine_forward,
    relu_backward,
    relu_forward,
    softmax_loss,
)
from cs231n.classifiers.fc_net import TwoLayerNet, FullyConnectedNet
from cs231n.gradient_check import eval_numerical_gradient


def rel_error(x, y):
    return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))


def test_data_loads():
    from cs231n.data_utils import load_CIFAR10

    Xtr, ytr, Xte, yte = load_CIFAR10(str(ROOT / "cs231n" / "datasets" / "cifar-10-batches-py"))
    assert Xtr.shape == (50000, 32, 32, 3)
    assert ytr.shape == (50000,)
    assert Xte.shape == (10000, 32, 32, 3)
    assert yte.shape == (10000,)


def test_knn_distances_and_labels():
    X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    y_train = np.array([0, 1, 2])
    X_test = np.array([[1.0, 1.0], [0.0, 1.5]])
    clf = KNearestNeighbor()
    clf.train(X_train, y_train)
    expected = np.array([[np.sqrt(2), 1.0, np.sqrt(2)], [1.5, np.sqrt(3.25), 0.5]])
    d2 = clf.compute_distances_two_loops(X_test)
    d1 = clf.compute_distances_one_loop(X_test)
    d0 = clf.compute_distances_no_loops(X_test)
    assert np.allclose(d2, expected)
    assert np.allclose(d1, expected)
    assert np.allclose(d0, expected)
    assert np.array_equal(clf.predict(X_test, k=1), np.array([1.0, 2.0]))


def test_softmax_vectorized_matches_naive():
    np.random.seed(231)
    W = 0.001 * np.random.randn(5, 4)
    X = np.random.randn(7, 5)
    y = np.array([0, 1, 2, 3, 1, 2, 0])
    loss_naive, grad_naive = softmax_loss_naive(W, X, y, 0.1)
    loss_vec, grad_vec = softmax_loss_vectorized(W, X, y, 0.1)
    assert abs(loss_naive - loss_vec) < 1e-12
    assert np.linalg.norm(grad_naive - grad_vec) < 1e-12
    assert np.linalg.norm(grad_vec) > 0


def test_affine_relu_softmax_layers():
    x = np.linspace(-0.1, 0.5, num=6).reshape(2, 3)
    w = np.linspace(-0.2, 0.3, num=12).reshape(3, 4)
    b = np.linspace(-0.3, 0.1, num=4)
    out, cache = affine_forward(x, w, b)
    assert out.shape == (2, 4)
    dout = np.random.randn(*out.shape)
    dx, dw, db = affine_backward(dout, cache)
    assert dx.shape == x.shape
    assert dw.shape == w.shape
    assert db.shape == b.shape
    relu_out, relu_cache = relu_forward(np.array([[-1.0, 0.5], [2.0, -0.1]]))
    assert np.array_equal(relu_out, np.array([[0.0, 0.5], [2.0, 0.0]]))
    assert np.array_equal(relu_backward(np.ones_like(relu_out), relu_cache), np.array([[0.0, 1.0], [1.0, 0.0]]))
    loss, dx_softmax = softmax_loss(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), np.array([2, 1]))
    assert loss > 0
    assert dx_softmax.shape == (2, 3)


def test_two_layer_net_gradients():
    np.random.seed(231)
    model = TwoLayerNet(input_dim=4, hidden_dim=5, num_classes=3, weight_scale=1e-2, reg=0.1)
    X = np.random.randn(6, 4)
    y = np.array([0, 1, 2, 2, 1, 0])
    loss, grads = model.loss(X, y)
    assert loss > 0
    for name in sorted(model.params):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
        assert rel_error(grad_num, grads[name]) < 1e-2, name


def test_fully_connected_net_forward_backward_and_training():
    np.random.seed(231)
    model = FullyConnectedNet([6, 5], input_dim=4, num_classes=3, weight_scale=1e-1, reg=0.1, dtype=np.float64)
    X = np.random.randn(5, 4)
    y = np.array([0, 1, 2, 2, 1])
    scores = model.loss(X)
    assert scores.shape == (5, 3)
    loss, grads = model.loss(X, y)
    assert loss > 0
    for name in sorted(model.params):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
        assert rel_error(grad_num, grads[name]) < 1e-2, name

    clf = Softmax()
    hist = clf.train(X, y, learning_rate=1e-1, reg=1e-3, num_iters=5, batch_size=4)
    assert len(hist) == 5
    assert clf.predict(X).shape == (5,)


if __name__ == "__main__":
    tests = [
        test_data_loads,
        test_knn_distances_and_labels,
        test_softmax_vectorized_matches_naive,
        test_affine_relu_softmax_layers,
        test_two_layer_net_gradients,
        test_fully_connected_net_forward_backward_and_training,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
