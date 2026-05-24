import os

import numpy as np
import scipy.io.wavfile


FS = 11025


def update_W(W, x, learning_rate):
    """
    Perform one stochastic gradient ascent update for ICA.

    The CS229 cocktail party assignment assumes Laplace-distributed independent
    sources, so the score term is sign(Wx).
    """
    source_estimate = W.dot(x)
    gradient = np.linalg.inv(W.T) - np.outer(np.sign(source_estimate), x)
    return W + learning_rate * gradient


def unmix(X, W):
    """
    Recover source signals from mixed signals with the learned unmixing matrix.
    """
    return X.dot(W.T)


def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))


def load_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mix_path = os.path.join(project_root, "data", "mix.dat")
    return np.loadtxt(mix_path)


def save_W(W):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, "W.txt"), W)


def save_sound(audio, name):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    scipy.io.wavfile.write(os.path.join(output_dir, f"{name}.wav"), FS, audio)


def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [
        0.1,
        0.1,
        0.1,
        0.05,
        0.05,
        0.05,
        0.02,
        0.02,
        0.01,
        0.01,
        0.005,
        0.005,
        0.002,
        0.002,
        0.001,
        0.001,
    ]
    print("Separating tracks ...")
    for lr in anneal:
        print(lr)
        rand = np.random.permutation(range(M))
        for i in rand:
            W = update_W(W, X[i], lr)

    return W


def main():
    np.random.seed(0)
    X = normalize(load_data())

    print("Mixed signal shape:", X.shape)
    for i in range(X.shape[1]):
        save_sound(X[:, i], f"mixed_{i}")

    W = unmixer(X)
    print("Learned W:")
    print(W)
    save_W(W)

    S = normalize(unmix(X, W))
    assert S.shape[1] == 5
    for i in range(S.shape[1]):
        save_sound(S[:, i], f"split_{i}")


if __name__ == "__main__":
    main()
