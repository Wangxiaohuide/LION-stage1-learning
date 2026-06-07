import json
from pathlib import Path

import numpy as np

from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.classifiers.linear_classifier import Softmax
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def flatten_with_bias(x):
    flat = np.reshape(x, (x.shape[0], -1))
    return np.hstack([flat, np.ones((flat.shape[0], 1))])


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


def run_knn(data):
    x_train = np.reshape(data["X_train"][:1000], (1000, -1))
    y_train = data["y_train"][:1000]
    x_val = np.reshape(data["X_val"][:200], (200, -1))
    y_val = data["y_val"][:200]

    classifier = KNearestNeighbor()
    classifier.train(x_train, y_train)

    results = {}
    for k in [1, 3, 5, 8]:
        y_pred = classifier.predict(x_val, k=k, num_loops=0)
        results[str(k)] = accuracy(y_pred, y_val)
    best_k = max(results, key=results.get)
    return {"validation_accuracy_by_k": results, "best_k": int(best_k), "best_val_acc": results[best_k]}


def run_softmax(data):
    x_train = flatten_with_bias(data["X_train"])
    x_val = flatten_with_bias(data["X_val"])
    y_train = data["y_train"]
    y_val = data["y_val"]

    search = {}
    best = {"val_acc": -1.0, "learning_rate": None, "reg": None}
    for lr in [1e-7, 5e-7, 1e-6]:
        for reg in [2.5e4, 5e4, 1e5]:
            classifier = Softmax()
            classifier.train(
                x_train,
                y_train,
                learning_rate=lr,
                reg=reg,
                num_iters=500,
                batch_size=200,
                verbose=False,
            )
            train_acc = accuracy(classifier.predict(x_train[:1000]), y_train[:1000])
            val_acc = accuracy(classifier.predict(x_val), y_val)
            key = f"lr={lr:g},reg={reg:g}"
            search[key] = {"train_acc": train_acc, "val_acc": val_acc}
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "learning_rate": lr, "reg": reg, "train_acc": train_acc}
    return {"search": search, "best": best}


def run_two_layer_net(data):
    small_data = {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
    }
    model = TwoLayerNet(
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-2,
        reg=0.5,
    )
    solver = Solver(
        model,
        small_data,
        update_rule="adam",
        optim_config={"learning_rate": 1e-3},
        lr_decay=0.95,
        num_epochs=3,
        batch_size=200,
        print_every=100,
        verbose=False,
        num_train_samples=1000,
        num_val_samples=1000,
    )
    solver.train()
    return {
        "best_val_acc": float(solver.best_val_acc),
        "last_train_acc": float(solver.train_acc_history[-1]),
        "last_val_acc": float(solver.val_acc_history[-1]),
        "final_loss": float(solver.loss_history[-1]),
        "num_loss_steps": len(solver.loss_history),
    }


def main():
    np.random.seed(231)
    data = get_CIFAR10_data(num_training=3000, num_validation=1000, num_test=1000)
    summary = {
        "dataset": {
            "X_train": list(data["X_train"].shape),
            "X_val": list(data["X_val"].shape),
            "X_test": list(data["X_test"].shape),
        },
        "knn": run_knn(data),
        "softmax_raw_pixels": run_softmax(data),
        "two_layer_net_raw_pixels": run_two_layer_net(data),
    }

    out_path = OUT / "week5_reproduction_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Week 5 CS231n Assignment 1 local reproduction")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
