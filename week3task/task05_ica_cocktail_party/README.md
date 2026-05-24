# Task05: ICA 鸡尾酒会问题

## 实验来源

- 课程：Stanford CS229
- 作业：Problem Set 4 - Independent Component Analysis
- 对应文件：`p04_ica.py`
- 任务主题：Cocktail Party Problem，即从多路混合音频中分离出原始声源

## 实验目标

本实验使用独立成分分析（Independent Component Analysis, ICA）处理 5 路混合音频。数据文件 `data/mix.dat` 中每一列是一条混合后的音轨，算法需要学习一个 unmixing matrix `W`，使得：

```text
S = X W^T
```

其中：

- `X` 是观测到的混合信号；
- `W` 是需要学习的解混矩阵；
- `S` 是分离后的独立声源估计。

## 核心实现

这次主要补全了两个函数。

### `update_W(W, x, learning_rate)`

对单个样本 `x` 做一次随机梯度上升更新。CS229 该题假设原始声源服从 Laplace 分布，因此梯度包含两部分：

```text
inv(W.T) - sign(Wx) x^T
```

代码实现为：

```python
source_estimate = W.dot(x)
gradient = np.linalg.inv(W.T) - np.outer(np.sign(source_estimate), x)
updated_W = W + learning_rate * gradient
```

### `unmix(X, W)`

用训练得到的 `W` 对所有混合信号做线性变换：

```python
S = X.dot(W.T)
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

## 学习收获

通过这个实验，我对无监督学习有了更具体的认识。ICA 没有使用人工标签，而是利用“不同声源之间统计独立”这一假设，从混合观测中恢复潜在结构。相比监督学习中用 `(x, y)` 训练模型，ICA 更像是在问：数据背后是否存在一些看不见但可以被分离出来的独立因素？

这也让我理解了鸡尾酒会问题的本质：如果多个麦克风同时接收到不同说话人和背景声音的线性混合，那么只要混合方式满足一定条件，就可以通过学习一个反向线性变换，把混合信号重新拆开。

本实验中最关键的代码不是很多，但公式和代码的对应关系很重要。`np.linalg.inv(W.T)` 来自 log determinant 项，`np.sign(W.dot(x))` 来自 Laplace 分布的对数密度梯度，`np.outer(...)` 则把单个样本上的梯度写成矩阵形式。
