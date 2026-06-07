"""
Run the CS231n Assignment 1 image-features experiment.

This covers the "Higher Level Representations: Image Features" part of A1:
HOG + HSV color histogram features are extracted from CIFAR-10 images, then
Softmax and TwoLayerNet are tuned on those features.
"""

import json
import time
from pathlib import Path

import numpy as np

from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.classifiers.linear_classifier import Softmax
from cs231n.data_utils import load_CIFAR10
from cs231n.features import color_histogram_hsv, extract_features, hog_feature
from cs231n.solver import Solver

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cs231n" / "datasets" / "cifar-10-batches-py"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

NUM_TRAINING = 49000
NUM_VAL = 1000
NUM_TEST = 1000


def load_raw_cifar10(seed=231):
    np.random.seed(seed)
    x_train, y_train, x_test, y_test = load_CIFAR10(str(DATA_DIR))

    mask = range(NUM_TRAINING, NUM_TRAINING + NUM_VAL)
    x_val = x_train[mask]
    y_val = y_train[mask]

    mask = range(NUM_TRAINING)
    x_train = x_train[mask]
    y_train = y_train[mask]

    mask = range(NUM_TEST)
    x_test = x_test[mask]
    y_test = y_test[mask]

    return {
        "X_train_raw": x_train,
        "y_train": y_train,
        "X_val_raw": x_val,
        "y_val": y_val,
        "X_test_raw": x_test,
        "y_test": y_test,
    }


def featurize(data):
    print("[features] extracting HOG + HSV color histogram")
    t0 = time.time()
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=10)]

    x_train_feats = extract_features(data["X_train_raw"], feature_fns, verbose=True)
    x_val_feats = extract_features(data["X_val_raw"], feature_fns)
    x_test_feats = extract_features(data["X_test_raw"], feature_fns)

    mean_feat = np.mean(x_train_feats, axis=0, keepdims=True)
    x_train_feats -= mean_feat
    x_val_feats -= mean_feat
    x_test_feats -= mean_feat

    std_feat = np.std(x_train_feats, axis=0, keepdims=True)
    std_feat[std_feat < 1e-12] = 1.0
    x_train_feats /= std_feat
    x_val_feats /= std_feat
    x_test_feats /= std_feat

    # Add bias dimension for linear classifiers, matching the assignment notebook.
    x_train_bias = np.hstack([x_train_feats, np.ones((x_train_feats.shape[0], 1))])
    x_val_bias = np.hstack([x_val_feats, np.ones((x_val_feats.shape[0], 1))])
    x_test_bias = np.hstack([x_test_feats, np.ones((x_test_feats.shape[0], 1))])

    wall = round(time.time() - t0, 2)
    info = {
        "feature_dim": int(x_train_feats.shape[1]),
        "feature_dim_with_bias": int(x_train_bias.shape[1]),
        "feature_functions": ["hog_feature", "color_histogram_hsv(nbin=10)"],
        "wall_seconds": wall,
    }
    print(f"[features] dim={info['feature_dim']} extraction_wall={wall}s")
    return {
        **data,
        "X_train_feats": x_train_feats,
        "X_val_feats": x_val_feats,
        "X_test_feats": x_test_feats,
        "X_train_bias": x_train_bias,
        "X_val_bias": x_val_bias,
        "X_test_bias": x_test_bias,
        "feature_info": info,
    }


def run_softmax_features(data):
    print("[features-softmax] tuning")
    t0 = time.time()
    learning_rates = [5e-8, 1e-7, 2.5e-7, 5e-7]
    regularization_strengths = [1e3, 2.5e3, 5e3, 1e4]
    search = {}
    best = {"val_acc": -1.0}

    for lr in learning_rates:
        for reg in regularization_strengths:
            clf = Softmax()
            clf.train(
                data["X_train_bias"],
                data["y_train"],
                learning_rate=lr,
                reg=reg,
                num_iters=1500,
                batch_size=256,
                verbose=False,
            )
            train_idx = np.random.choice(NUM_TRAINING, 3000, replace=False)
            train_acc = float(np.mean(clf.predict(data["X_train_bias"][train_idx]) == data["y_train"][train_idx]))
            val_acc = float(np.mean(clf.predict(data["X_val_bias"]) == data["y_val"]))
            key = f"lr={lr:g},reg={reg:g}"
            search[key] = {
                "learning_rate": lr,
                "reg": reg,
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
            }
            print(f"  {key} train={train_acc:.4f} val={val_acc:.4f}")
            if val_acc > best["val_acc"]:
                best = {
                    "learning_rate": lr,
                    "reg": reg,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "classifier": clf,
                }

    test_acc = float(np.mean(best["classifier"].predict(data["X_test_bias"]) == data["y_test"]))
    result = {
        "model": "Softmax",
        "representation": "HOG+HSV color histogram",
        "num_training": NUM_TRAINING,
        "num_validation": NUM_VAL,
        "num_test": NUM_TEST,
        "search": search,
        "best": {
            "learning_rate": best["learning_rate"],
            "reg": best["reg"],
            "train_acc": round(best["train_acc"], 4),
            "val_acc": round(best["val_acc"], 4),
            "test_acc": round(test_acc, 4),
        },
        "wall_seconds": round(time.time() - t0, 2),
    }
    (OUT / "features_softmax_49k.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[features-softmax] best val={best['val_acc']:.4f} test={test_acc:.4f}")
    return result


def run_twolayer_features(data):
    print("[features-twolayer] tuning")
    t0 = time.time()
    configs = [
        {"hidden_dim": 250, "learning_rate": 5e-4, "reg": 0.001, "num_epochs": 8},
        {"hidden_dim": 500, "learning_rate": 5e-4, "reg": 0.001, "num_epochs": 8},
        {"hidden_dim": 500, "learning_rate": 1e-3, "reg": 0.001, "num_epochs": 8},
        {"hidden_dim": 500, "learning_rate": 1e-3, "reg": 0.01, "num_epochs": 8},
        {"hidden_dim": 750, "learning_rate": 5e-4, "reg": 0.001, "num_epochs": 8},
        {"hidden_dim": 750, "learning_rate": 1e-3, "reg": 0.001, "num_epochs": 8},
    ]
    results = []
    best_solver = None
    best_entry = None

    solver_data = {
        "X_train": data["X_train_feats"],
        "y_train": data["y_train"],
        "X_val": data["X_val_feats"],
        "y_val": data["y_val"],
    }

    for i, cfg in enumerate(configs, start=1):
        np.random.seed(231 + i)
        model = TwoLayerNet(
            input_dim=data["X_train_feats"].shape[1],
            hidden_dim=cfg["hidden_dim"],
            num_classes=10,
            weight_scale=1e-2,
            reg=cfg["reg"],
        )
        solver = Solver(
            model,
            solver_data,
            update_rule="adam",
            optim_config={"learning_rate": cfg["learning_rate"]},
            lr_decay=0.95,
            num_epochs=cfg["num_epochs"],
            batch_size=256,
            print_every=9999,
            verbose=False,
            num_train_samples=3000,
            num_val_samples=None,
        )
        c0 = time.time()
        solver.train()
        scores = model.loss(data["X_test_feats"])
        test_acc = float(np.mean(np.argmax(scores, axis=1) == data["y_test"]))
        entry = {
            "cfg_id": i,
            **cfg,
            "batch_size": 256,
            "best_val_acc": round(float(solver.best_val_acc), 4),
            "last_train_acc": round(float(solver.train_acc_history[-1]), 4),
            "last_val_acc": round(float(solver.val_acc_history[-1]), 4),
            "test_acc": round(test_acc, 4),
            "final_loss": round(float(solver.loss_history[-1]), 6),
            "wall_seconds": round(time.time() - c0, 2),
        }
        results.append(entry)
        print(
            f"  cfg{i} H={cfg['hidden_dim']} lr={cfg['learning_rate']:g} "
            f"reg={cfg['reg']} val={entry['best_val_acc']:.4f} test={entry['test_acc']:.4f}"
        )
        if best_entry is None or entry["best_val_acc"] > best_entry["best_val_acc"]:
            best_entry = entry
            best_solver = solver

    loss_history = [float(x) for x in best_solver.loss_history]
    result = {
        "model": "TwoLayerNet",
        "representation": "HOG+HSV color histogram",
        "num_training": NUM_TRAINING,
        "num_validation": NUM_VAL,
        "num_test": NUM_TEST,
        "all_configs": results,
        "best": best_entry,
        "best_loss_history": loss_history,
        "wall_seconds": round(time.time() - t0, 2),
    }
    (OUT / "features_twolayer_49k.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[features-twolayer] best cfg={best_entry['cfg_id']} val={best_entry['best_val_acc']:.4f} test={best_entry['test_acc']:.4f}")
    return result


def main():
    total_t0 = time.time()
    raw = load_raw_cifar10()
    data = featurize(raw)
    softmax = run_softmax_features(data)
    twolayer = run_twolayer_features(data)
    summary = {
        "scale": "N49000",
        "representation": "HOG+HSV color histogram",
        "feature_info": data["feature_info"],
        "softmax": softmax,
        "two_layer_net": twolayer,
        "total_wall_seconds": round(time.time() - total_t0, 2),
    }
    (OUT / "week5_features_reproduction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[features] wrote results/week5_features_reproduction_summary.json")


if __name__ == "__main__":
    main()
