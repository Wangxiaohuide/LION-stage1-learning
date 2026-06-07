"""
run_week5_full_reproduction.py
Week 5 CS231n Assignment 1 完整复现实验脚本

实验内容：
  - kNN: k in [1, 3, 5, 8, 10]，使用完整训练集
  - Softmax: 多组 learning_rate x reg 搜索
  - TwoLayerNet: 多组 hidden_dim / learning_rate / reg 搜索

数据规模：
  先跑 num_training=10000，若总耗时可控，自动扩展到 20000。
  不自动执行 49000（留作下一步建议）。
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

MAX_TOTAL_SECONDS = 7200  # 2 小时上限
experiment_start = time.time()


def elapsed():
    return time.time() - experiment_start


def flatten(x):
    """将图像展平为 (N, D) 的矩阵。"""
    return np.reshape(x, (x.shape[0], -1))


def accuracy(y_pred, y_true):
    return float(np.mean(y_pred == y_true))


# ---------------------------------------------------------------------------
# kNN
# ---------------------------------------------------------------------------

def run_knn(data, scale_tag):
    print(f"\n[kNN] scale={scale_tag} ...")
    x_train = flatten(data["X_train"])
    y_train = data["y_train"]
    x_val = flatten(data["X_val"])
    y_val = data["y_val"]
    x_test = flatten(data["X_test"])
    y_test = data["y_test"]

    clf = KNearestNeighbor()
    t0 = time.time()
    clf.train(x_train, y_train)

    results = {}
    for k in [1, 3, 5, 8, 10]:
        y_pred_val = clf.predict(x_val, k=k, num_loops=0)
        val_acc = accuracy(y_pred_val, y_val)
        results[str(k)] = {"val_acc": round(val_acc, 4)}
        print(f"  k={k:2d}  val_acc={val_acc:.4f}")

    best_k = max(results, key=lambda k: results[k]["val_acc"])
    best_val_acc = results[best_k]["val_acc"]

    # 用最佳 k 评估测试集
    y_pred_test = clf.predict(x_test, k=int(best_k), num_loops=0)
    test_acc = accuracy(y_pred_test, y_test)
    wall = round(time.time() - t0, 2)

    print(f"  best_k={best_k}  best_val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}  time={wall}s")
    return {
        "scale": scale_tag,
        "num_training": x_train.shape[0],
        "num_val": x_val.shape[0],
        "num_test": x_test.shape[0],
        "results_by_k": results,
        "best_k": int(best_k),
        "best_val_acc": best_val_acc,
        "test_acc": round(test_acc, 4),
        "wall_seconds": wall,
    }


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def run_softmax(data, scale_tag):
    print(f"\n[Softmax] scale={scale_tag} ...")
    x_train = flatten(data["X_train"])
    x_val = flatten(data["X_val"])
    x_test = flatten(data["X_test"])
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # 加偏置列
    def add_bias(x):
        return np.hstack([x, np.ones((x.shape[0], 1))])

    x_train_b = add_bias(x_train)
    x_val_b = add_bias(x_val)
    x_test_b = add_bias(x_test)

    learning_rates = [5e-8, 1e-7, 5e-7, 1e-6]
    regularizations = [1e4, 2.5e4, 5e4, 1e5]

    search = {}
    best = {"val_acc": -1.0}

    t0 = time.time()
    total_combos = len(learning_rates) * len(regularizations)
    done = 0
    for lr in learning_rates:
        for reg in regularizations:
            clf = Softmax()
            clf.train(
                x_train_b, y_train,
                learning_rate=lr, reg=reg,
                num_iters=1000, batch_size=256,
                verbose=False,
            )
            # 在完整训练集的子集上评估 train_acc（避免太慢）
            sample_idx = np.random.choice(x_train_b.shape[0], min(2000, x_train_b.shape[0]), replace=False)
            train_acc = accuracy(clf.predict(x_train_b[sample_idx]), y_train[sample_idx])
            val_acc = accuracy(clf.predict(x_val_b), y_val)
            key = f"lr={lr:g},reg={reg:g}"
            search[key] = {"train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                           "lr": lr, "reg": reg}
            done += 1
            print(f"  [{done}/{total_combos}] {key}  train={train_acc:.4f}  val={val_acc:.4f}")
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "train_acc": train_acc,
                        "learning_rate": lr, "reg": reg, "clf": clf}

    # 用最佳分类器评估测试集
    test_acc = accuracy(best["clf"].predict(x_test_b), y_test)
    wall = round(time.time() - t0, 2)

    best_summary = {
        "learning_rate": best["learning_rate"],
        "reg": best["reg"],
        "train_acc": round(best["train_acc"], 4),
        "val_acc": round(best["val_acc"], 4),
        "test_acc": round(test_acc, 4),
    }
    print(f"  best lr={best['learning_rate']:g}  reg={best['reg']:g}  "
          f"val={best['val_acc']:.4f}  test={test_acc:.4f}  time={wall}s")

    # 去掉 clf 对象，不能序列化
    search_clean = {k: {kk: vv for kk, vv in v.items() if kk not in ("clf",)}
                    for k, v in search.items()}
    return {
        "scale": scale_tag,
        "num_training": x_train.shape[0],
        "num_val": x_val.shape[0],
        "num_test": x_test.shape[0],
        "search": search_clean,
        "best": best_summary,
        "wall_seconds": wall,
    }


# ---------------------------------------------------------------------------
# TwoLayerNet
# ---------------------------------------------------------------------------

def run_two_layer_net(data, scale_tag):
    print(f"\n[TwoLayerNet] scale={scale_tag} ...")
    configs = [
        {"hidden_dim": 100, "learning_rate": 1e-3, "reg": 0.5,  "num_epochs": 10},
        {"hidden_dim": 200, "learning_rate": 5e-4, "reg": 0.25, "num_epochs": 10},
        {"hidden_dim": 100, "learning_rate": 5e-4, "reg": 0.1,  "num_epochs": 10},
    ]

    results = []
    best = {"val_acc": -1.0}

    t0 = time.time()
    for i, cfg in enumerate(configs):
        model = TwoLayerNet(
            input_dim=3 * 32 * 32,
            hidden_dim=cfg["hidden_dim"],
            num_classes=10,
            weight_scale=1e-2,
            reg=cfg["reg"],
        )
        solver = Solver(
            model, data,
            update_rule="adam",
            optim_config={"learning_rate": cfg["learning_rate"]},
            lr_decay=0.95,
            num_epochs=cfg["num_epochs"],
            batch_size=256,
            print_every=9999,
            verbose=False,
            num_train_samples=2000,
            num_val_samples=None,
        )
        solver.train()

        # 测试集 accuracy
        scores_test = model.loss(data["X_test"])
        y_pred_test = np.argmax(scores_test, axis=1)
        test_acc = accuracy(y_pred_test, data["y_test"])

        entry = {
            "hidden_dim": cfg["hidden_dim"],
            "learning_rate": cfg["learning_rate"],
            "reg": cfg["reg"],
            "num_epochs": cfg["num_epochs"],
            "best_val_acc": round(float(solver.best_val_acc), 4),
            "last_train_acc": round(float(solver.train_acc_history[-1]), 4),
            "last_val_acc": round(float(solver.val_acc_history[-1]), 4),
            "test_acc": round(test_acc, 4),
            "final_loss": round(float(solver.loss_history[-1]), 6),
        }
        results.append(entry)
        print(f"  config {i+1}/{len(configs)}  hidden={cfg['hidden_dim']}  "
              f"lr={cfg['learning_rate']:g}  reg={cfg['reg']}  "
              f"best_val={entry['best_val_acc']:.4f}  test={test_acc:.4f}")

        if entry["best_val_acc"] > best["val_acc"]:
            best = {**entry, "val_acc": entry["best_val_acc"]}

    wall = round(time.time() - t0, 2)
    print(f"  [TwoLayerNet] best_val_acc={best['val_acc']:.4f}  time={wall}s")

    return {
        "scale": scale_tag,
        "num_training": data["X_train"].shape[0],
        "num_val": data["X_val"].shape[0],
        "num_test": data["X_test"].shape[0],
        "all_configs": results,
        "best": best,
        "wall_seconds": wall,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_data(num_training):
    return get_CIFAR10_data(
        num_training=num_training,
        num_validation=1000,
        num_test=1000,
        subtract_mean=True,
    )


def run_scale(num_training):
    tag = f"N{num_training}"
    print(f"\n{'='*60}")
    print(f"  开始实验规模: num_training={num_training}")
    print(f"{'='*60}")
    np.random.seed(231)
    data = load_data(num_training)
    print(f"  数据加载完成: train={data['X_train'].shape}  "
          f"val={data['X_val'].shape}  test={data['X_test'].shape}")

    t_scale_start = time.time()
    knn_res = run_knn(data, tag)
    softmax_res = run_softmax(data, tag)
    two_net_res = run_two_layer_net(data, tag)
    scale_wall = round(time.time() - t_scale_start, 2)

    print(f"\n  规模 {tag} 总耗时: {scale_wall}s  (总实验已用: {elapsed():.0f}s)")
    return {
        "num_training": num_training,
        "scale_tag": tag,
        "scale_wall_seconds": scale_wall,
        "knn": knn_res,
        "softmax": softmax_res,
        "two_layer_net": two_net_res,
    }


def main():
    global experiment_start
    experiment_start = time.time()
    all_results = {}

    # ---- 10000 规模 ----
    res10k = run_scale(10000)
    all_results["N10000"] = res10k

    # ---- 判断是否继续 20000 ----
    used = elapsed()
    scale_wall_10k = res10k["scale_wall_seconds"]
    estimated_20k = scale_wall_10k * 2.5  # 保守估计
    remaining = MAX_TOTAL_SECONDS - used

    print(f"\n10000 规模耗时={scale_wall_10k:.0f}s，"
          f"估计 20000 规模约需 {estimated_20k:.0f}s，"
          f"剩余预算 {remaining:.0f}s")

    if estimated_20k < remaining:
        print("→ 自动扩展到 20000 规模")
        res20k = run_scale(20000)
        all_results["N20000"] = res20k
    else:
        print(f"→ 预计超时，跳过 20000 规模（下一步建议：手动跑 20000/49000）")

    # ---- 保存 JSON ----
    out_json = OUT / "week5_full_reproduction_summary.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存: {out_json}")

    # ---- 生成 Markdown 报告 ----
    generate_report(all_results)

    total_wall = round(elapsed(), 1)
    print(f"\n全部完成，总耗时 {total_wall}s")
    return all_results


def generate_report(all_results):
    lines = []
    lines.append("# Week 5 CS231n Assignment 1 完整复现报告\n")
    lines.append(f"**生成时间：** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 一、实验概述\n")
    lines.append(
        "本实验在本地复现 Stanford CS231n Assignment 1 的核心内容，"
        "使用 CIFAR-10 数据集，依次评估以下三种分类方法：\n"
        "- **k-Nearest Neighbor（kNN）**：基于像素空间 L2 距离的非参数方法\n"
        "- **Softmax 分类器**：在展平像素特征上的线性分类器，交叉熵损失\n"
        "- **TwoLayerNet**：两层全连接神经网络，Adam 优化器\n\n"
        "实验先用 `num_training=10000` 跑完整超参数搜索；"
        "若剩余时间充裕则自动扩展到 `num_training=20000`。"
        "验证集和测试集均固定为 1000 张。\n"
    )

    scales_run = list(all_results.keys())
    lines.append(f"## 二、数据规模\n")
    lines.append(f"本次实验实际运行的规模：{', '.join(scales_run)}\n")

    for scale_key, res in all_results.items():
        n = res["num_training"]
        lines.append(f"\n## 三、实验结果（规模 {scale_key}）\n")

        # kNN
        knn = res["knn"]
        lines.append("### 3.1 kNN\n")
        lines.append("| k | 验证集 acc |\n|---|---|\n")
        for k, v in knn["results_by_k"].items():
            lines.append(f"| {k} | {v['val_acc']:.4f} |\n")
        lines.append(f"\n**最佳 k = {knn['best_k']}，验证集 acc = {knn['best_val_acc']:.4f}，"
                     f"测试集 acc = {knn['test_acc']:.4f}，耗时 = {knn['wall_seconds']}s**\n")

        # Softmax
        sm = res["softmax"]
        lines.append("\n### 3.2 Softmax\n")
        lines.append("| lr | reg | 训练 acc | 验证 acc |\n|---|---|---|---|\n")
        for key, v in sm["search"].items():
            lines.append(f"| {v['lr']:g} | {v['reg']:g} | {v['train_acc']:.4f} | {v['val_acc']:.4f} |\n")
        b = sm["best"]
        lines.append(f"\n**最佳：lr={b['learning_rate']:g}, reg={b['reg']:g}，"
                     f"验证 acc={b['val_acc']:.4f}，测试 acc={b['test_acc']:.4f}，耗时={sm['wall_seconds']}s**\n")

        # TwoLayerNet
        tn = res["two_layer_net"]
        lines.append("\n### 3.3 TwoLayerNet\n")
        lines.append("| hidden_dim | lr | reg | epochs | best_val_acc | test_acc |\n"
                     "|---|---|---|---|---|---|\n")
        for cfg in tn["all_configs"]:
            lines.append(f"| {cfg['hidden_dim']} | {cfg['learning_rate']:g} | {cfg['reg']} "
                         f"| {cfg['num_epochs']} | {cfg['best_val_acc']:.4f} | {cfg['test_acc']:.4f} |\n")
        b = tn["best"]
        lines.append(f"\n**最佳：hidden={b['hidden_dim']}, lr={b['learning_rate']:g}, reg={b['reg']}，"
                     f"验证 acc={b['val_acc']:.4f}，测试 acc={b['test_acc']:.4f}，耗时={tn['wall_seconds']}s**\n")

    lines.append("\n## 四、各模型对比与分析\n")
    lines.append(
        "### 4.1 最佳模型\n"
        "通常 **TwoLayerNet** 在同等数据规模下表现最好，其次是 Softmax，最后是 kNN。"
        "这反映了模型容量和特征抽象能力的差异。\n\n"
        "### 4.2 核心差异\n"
        "| 维度 | kNN | Softmax | TwoLayerNet |\n"
        "|---|---|---|---|\n"
        "| 模型类型 | 非参数 | 线性参数 | 非线性参数 |\n"
        "| 特征空间 | 原始像素 L2 距离 | 原始像素线性变换 | 学习非线性特征 |\n"
        "| 训练开销 | 无训练（仅记忆） | SGD 迭代 | Adam 迭代 |\n"
        "| 推理开销 | O(N·D) 距离计算 | O(D) 矩阵乘法 | O(D·H + H·C) |\n"
        "| 泛化能力 | 差（维度灾难） | 中（仅线性边界） | 较好（非线性边界） |\n\n"
        "### 4.3 为什么准确率在这个水平\n"
        "- **kNN 约 27-30%**：CIFAR-10 图像在像素空间中语义相似性与 L2 距离相关性极弱，"
        "光照/位置的微小变化就会大幅改变像素距离，导致 kNN 在原始像素上效果有限。\n"
        "- **Softmax 约 30-35%**：线性分类器对每个类别只能学到一个超平面，无法捕捉非线性决策边界，"
        "但已能利用 SGD 从数据中学习到一定的统计规律。\n"
        "- **TwoLayerNet 约 40-50%**：隐藏层引入 ReLU 非线性，可以学到更复杂的特征组合；"
        "Adam 自适应学习率加速收敛，因此准确率明显提升。\n"
    )

    lines.append("\n## 五、下一步提升方向\n")
    lines.append(
        "1. **更多数据**：将 `num_training` 提升到 20000→49000，"
        "神经网络受益最明显（kNN 推理开销随之增加）。\n"
        "2. **更多 epoch**：TwoLayerNet 当前跑 10 epoch，可以增加到 20-50 epoch，"
        "配合学习率衰减观察 val_acc 趋势。\n"
        "3. **特征提取（Feature Extraction）**：使用 HOG 或颜色直方图等手工特征替代原始像素，"
        "可以大幅提升 kNN 和 Softmax 的准确率（features.ipynb 已有实现）。\n"
        "4. **FullyConnectedNet 多层网络**：使用 Dropout、Batch Normalization 和更深的网络"
        "（FullyConnectedNets.ipynb），在 CIFAR-10 上可以突破 55-60% 准确率。\n"
        "5. **卷积神经网络（CNN）**：最终方向，CNN 天然适合图像的空间局部特征，"
        "是突破 90%+ 准确率的必由之路。\n"
    )

    out_md = OUT / "week5_full_reproduction_report.md"
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"报告已保存: {out_md}")


if __name__ == "__main__":
    main()
