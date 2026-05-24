# Task05: CS229 ICA 鸡尾酒会问题

## 实验背景

本实验对应 Stanford CS229 Problem Set 4 中的 Independent Component Analysis（ICA）编程题。它和本周学习的第 15 讲内容直接相关：

- Lecture 13：K-means、Mixture of Gaussians、EM
- Lecture 14：Factor Analysis
- Lecture 15：PCA、ICA
- Lecture 16：MDP、Bellman Equations
- Lecture 17：Value Iteration、Policy Iteration、LQR、LQG

本实验重点实践 Lecture 15 中的 ICA，用来解决经典的 Cocktail Party Problem：从多路混合音频中分离出原始独立声源。

## 实验目标

数据文件 `data/mix.dat` 包含 5 路混合音频。每一列是一条混合后的音轨，每一行对应一个时间点。实验目标是学习一个解混矩阵 `W`，使混合信号 `X` 经过线性变换后得到分离声源 `S`：

```text
S = X W^T
```

其中：

- `X` 是观测到的混合信号；
- `W` 是 ICA 学习得到的 unmixing matrix；
- `S` 是分离后的独立声源估计。

## 核心实现

本次主要补全 `src/p04_ica.py` 中的两个函数。

### `update_W(W, x, learning_rate)`

对单个样本 `x` 做一次随机梯度上升更新。CS229 该题假设原始声源服从 Laplace 分布，因此更新梯度包含：

```text
inv(W.T) - sign(Wx) x^T
```

代码实现为：

```python
def update_W(W, x, learning_rate):
    source_estimate = W.dot(x)
    gradient = np.linalg.inv(W.T) - np.outer(np.sign(source_estimate), x)
    return W + learning_rate * gradient
```

我的理解：

- `W.dot(x)` 是当前估计出的独立声源；
- `np.sign(W.dot(x))` 来自 Laplace 分布的对数密度梯度；
- `np.linalg.inv(W.T)` 来自 log determinant 项；
- `np.outer(...)` 将单个样本上的梯度写成矩阵形式。

### `unmix(X, W)`

用学习到的 `W` 对所有混合信号做解混：

```python
def unmix(X, W):
    return X.dot(W.T)
```

## 运行方式

在当前任务目录下运行：

```bash
python src/p04_ica.py
```

程序会在 `src/output/` 中生成：

- `mixed_0.wav` 至 `mixed_4.wav`：原始混合音频；
- `split_0.wav` 至 `split_4.wav`：ICA 分离后的音频；
- `W.txt`：学习到的解混矩阵。

## 本地运行记录

本地运行时，数据形状为：

```text
Mixed signal shape: (53442, 5)
```

这表示一共有 53442 个时间点、5 路混合信号。经过退火学习率训练后，程序能够输出 5 条分离音频。

## 学习收获

这个实验帮助我把 ICA 的公式和代码联系起来。相比 K-means、GMM、PCA 等无监督学习方法，ICA 更强调“独立性”假设，而不是只关注距离、概率成分或方差方向。

鸡尾酒会问题也让我更直观地理解了无监督学习的价值：即使没有人工标签，只要我们对数据生成过程有合理假设，比如不同声源之间相互独立，就可以从混合观测中恢复隐藏结构。

本实验目前还需要继续加强的地方是 ICA 更新公式的完整推导，尤其是 log determinant 项和 Laplace 分布假设如何共同得到最终梯度。
