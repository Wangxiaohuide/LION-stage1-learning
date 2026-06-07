"""
run_week5_49000_reproduction.py
Week 5 CS231n Assignment 1 — 49000 规模完整复现脚本

官方常用数据规模: num_training=49000, num_validation=1000, num_test=1000

实验内容:
  A. Softmax raw pixels — 16 组超参数搜索
  B. TwoLayerNet raw pixels — 8 组代表性配置
  C. kNN — k=[5, 8, 10] 简化版（49000 规模推理较慢）

注意: 由于 bash 工具单次调用 45s 超时限制，本脚本被拆分为独立函数，
      主流程分段在 bash 中执行，结果保存为 JSON 后合并。
"""

import json
import time
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

NUM_TRAINING = 49000
NUM_VAL = 1000
NUM_TEST = 1000


def load_data(seed=231):
    np.random.seed(seed)
    return get_CIFAR10_data(
        num_training=NUM_TRAINING,
        num_validation=NUM_VAL,
        num_test=NUM_TEST,
        subtract_mean=True,
    )


def run_knn(data):
    """kNN 简化版: k=[5, 8, 10]"""
    print("\n[kNN] N49000, k=[5,8,10]")
    x_tr = data["X_train"].reshape(NUM_TRAINING, -1)
    x_v = data["X_val"].reshape(NUM_VAL, -1)
    x_t = data["X_test"].reshape(NUM_TEST, -1)

    clf = KNearestNeighbor()
    clf.train(x_tr, data["y_train"])

    t0 = time.time()
    results = {}
    for k in [5, 8, 10]:
        tk = time.time()
        yp = clf.predict(x_v, k=k, num_loops=0)
        va = float(np.mean(yp == data["y_val"]))
        results[str(k)] = {"val_acc": round(va, 4)}
        print(f"  k={k} val_acc={va:.4f}  t={time.time()-tk:.1f}s")

    best_k = max(results, key=lambda k: results[k]["val_acc"])
    yp_t = clf.predict(x_t, k=int(best_k), num_loops=0)
    te = float(np.mean(yp_t == data["y_test"]))
    wall = round(time.time() - t0, 2)

    print(f"  best_k={best_k}  test_acc={te:.4f}  wall={wall}s")
    res = {
        "scale": "N49000",
        "num_training": NUM_TRAINING,
        "ks_tested": [5, 8, 10],
        "results_by_k": results,
        "best_k": int(best_k),
        "best_val_acc": results[best_k]["val_acc"],
        "test_acc": round(te, 4),
        "wall_seconds": wall,
        "note": "只测 k=[5,8,10]，49000 规模推理每个 k 约 12s",
    }
    (OUT / "knn_49k.json").write_text(json.dumps(res, indent=2))
    return res


def run_softmax_batch(data, batch_id, lrs, regs, num_iters=1000, batch_size=256):
    """Softmax 超参搜索，分批运行"""
    print(f"\n[Softmax] N49000 batch={batch_id}")
    x_tr = data["X_train"].reshape(NUM_TRAINING, -1)
    x_v = data["X_val"].reshape(NUM_VAL, -1)
    x_t = data["X_test"].reshape(NUM_TEST, -1)

    def ab(x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    xtr, xv, xt = ab(x_tr), ab(x_v), ab(x_t)
    y_tr, y_v, y_t = data["y_train"], data["y_val"], data["y_test"]

    search = {}
    best = {"val_acc": -1.0}
    total = len(lrs) * len(regs)
    done = 0
    t0 = time.time()

    for lr in lrs:
        for reg in regs:
            clf = Softmax()
            clf.train(xtr, y_tr, learning_rate=lr, reg=reg,
                      num_iters=num_iters, batch_size=batch_size, verbose=False)
            idx = np.random.choice(NUM_TRAINING, 2000, replace=False)
            ta = float(np.mean(clf.predict(xtr[idx]) == y_tr[idx]))
            va = float(np.mean(clf.predict(xv) == y_v))
            key = f"lr={lr:g},reg={reg:g}"
            search[key] = {"lr": lr, "reg": reg,
                           "train_acc": round(ta, 4), "val_acc": round(va, 4)}
            done += 1
            print(f"  [{done}/{total}] {key}  train={ta:.4f} val={va:.4f}  t={time.time()-t0:.0f}s")
            if va > best["val_acc"]:
                best = {"val_acc": va, "train_acc": ta, "lr": lr, "reg": reg, "clf": clf}

    te = float(np.mean(best["clf"].predict(xt) == y_t))
    wall = round(time.time() - t0, 2)
    print(f"  batch best: lr={best['lr']:g} reg={best['reg']:g} val={best['val_acc']:.4f} test={te:.4f}  {wall}s")

    res = {
        "batch_id": batch_id,
        "search": search,
        "batch_best": {
            "learning_rate": best["lr"], "reg": best["reg"],
            "train_acc": round(best["train_acc"], 4),
            "val_acc": round(best["val_acc"], 4),
            "test_acc": round(te, 4),
        },
        "wall_seconds": wall,
    }
    (OUT / f"softmax_49k_batch{batch_id}.json").write_text(json.dumps(res, indent=2))
    return res


def run_twolayer_config(data, cfg_id, hidden_dim, lr, reg, num_epochs, batch_size=512, seed=231):
    """单个 TwoLayerNet 配置"""
    np.random.seed(seed)
    print(f"\n[TwoLayerNet] N49000 cfg={cfg_id} hidden={hidden_dim} lr={lr:g} reg={reg} epochs={num_epochs}")
    t0 = time.time()

    model = TwoLayerNet(input_dim=3 * 32 * 32, hidden_dim=hidden_dim,
                        num_classes=10, weight_scale=1e-2, reg=reg)
    solver = Solver(model, data,
                    update_rule="adam",
                    optim_config={"learning_rate": lr},
                    lr_decay=0.95,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    print_every=9999,
                    verbose=False,
                    num_train_samples=1000,
                    num_val_samples=None)
    solver.train()

    sc = model.loss(data["X_test"])
    te = float(np.mean(np.argmax(sc, axis=1) == data["y_test"]))
    wall = round(time.time() - t0, 2)

    entry = {
        "cfg_id": cfg_id,
        "hidden_dim": hidden_dim,
        "learning_rate": lr,
        "reg": reg,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "best_val_acc": round(float(solver.best_val_acc), 4),
        "last_train_acc": round(float(solver.train_acc_history[-1]), 4),
        "last_val_acc": round(float(solver.val_acc_history[-1]), 4),
        "test_acc": round(te, 4),
        "final_loss": round(float(solver.loss_history[-1]), 6),
        "wall_seconds": wall,
    }
    print(f"  best_val={entry['best_val_acc']:.4f}  test={te:.4f}  {wall}s")
    (OUT / f"tn49_cfg{cfg_id}.json").write_text(json.dumps(entry, indent=2))
    return entry


if __name__ == "__main__":
    """直接运行时跑全流程（但每部分建议分开运行以避免超时）"""
    data = load_data()
    run_knn(data)
    run_softmax_batch(data, 1, [2.5e-7, 5e-7], [5e3, 1e4, 2.5e4, 5e4])
    run_softmax_batch(data, 2, [7.5e-7, 1e-6], [5e3, 1e4, 2.5e4, 5e4])
