# 第三周学习报告：CS229 第 13-17 讲与 ICA 鸡尾酒会实验

## 一、本周学习概况

本周主要学习 Stanford CS229 第 13-17 讲，内容从无监督学习逐步过渡到强化学习与控制，并完成了 Problem Set 4 中的 Independent Component Analysis（ICA）鸡尾酒会编程作业。

根据本地下载的 `cs229/syllabus-autumn2018.html`，第 13-17 讲对应内容为：

| 讲次 | 主题 | 本周理解重点 |
|---|---|---|
| Lecture 13 | K-means, Mixture of Gaussians, Expectation Maximization | 从无标签数据中发现聚类结构，理解硬分配和软分配的区别 |
| Lecture 14 | Factor Analysis | 用潜变量解释高维观测数据，理解数据背后的低维生成因素 |
| Lecture 15 | Principal Component Analysis, Independent Component Analysis | 学习 PCA 降维和 ICA 独立成分分离，并完成鸡尾酒会实验 |
| Lecture 16 | MDPs, Bellman Equations | 建立强化学习中的状态、动作、奖励、转移概率和价值函数框架 |
| Lecture 17 | Value Iteration, Policy Iteration, LQR, LQG | 学习如何通过迭代方法求解最优策略，并初步接触控制问题 |

这一周的学习重点不再是给定标签后训练分类器，而是理解数据自身结构、潜变量建模、独立信号分离，以及智能体如何在环境中做长期决策。

## 二、Lecture 13：K-means、GMM 与 EM

K-means 是无监督学习中最直观的聚类方法。它的核心目标是在没有标签的情况下，把样本划分成 `K` 个簇，使同一簇中的样本尽量相似，不同簇之间尽量分开。

它的流程可以总结为：

1. 初始化 `K` 个聚类中心；
2. 将每个样本分配给最近的中心；
3. 根据当前分配结果重新计算每个簇的中心；
4. 重复上述步骤，直到聚类结果基本稳定。

我目前对 K-means 的理解是：它是一种“硬分配”的聚类方法，每个样本最终只属于一个簇。它简单直观，但也容易受到初始中心、特征尺度和 `K` 的选择影响。

Mixture of Gaussians（GMM）比 K-means 更柔性。GMM 假设数据来自多个高斯分布的混合，每个样本不是简单地被分到某一类，而是具有属于每个高斯成分的概率。这让我理解到聚类也可以带有不确定性，而不是只能给出一个绝对判断。

EM 算法用于处理存在隐变量的问题。它的基本思想是：

- E-step：根据当前参数估计隐变量的后验概率；
- M-step：根据隐变量的估计结果更新模型参数。

这一讲让我认识到，无监督学习中很多任务的困难在于“真正决定数据来源的变量是看不见的”。EM 不是直接一次性求解所有未知量，而是在估计隐变量和更新参数之间反复迭代。

## 三、Lecture 14：Factor Analysis

Factor Analysis（因子分析）关注的是：高维观测数据背后是否存在少量低维潜在因素。

它的基本思想可以理解为：

```text
x = Lambda z + u
```

其中：

- `z` 是低维潜变量；
- `Lambda` 描述潜变量如何影响观测变量；
- `u` 是噪声；
- `x` 是最终观测到的数据。

这一讲让我开始理解“潜变量建模”的意义。很多高维数据表面上维度很高，但真正影响数据变化的因素可能并不多。例如一个人的多项测量指标，背后可能由健康状况、体型、运动习惯等少数因素共同影响。

Factor Analysis 和后面 PCA、ICA 的联系在于，它们都在尝试从观测数据中找到更底层、更简洁的结构。不同的是，Factor Analysis 更强调概率生成模型和噪声假设。

## 四、Lecture 15：PCA、ICA 与鸡尾酒会问题

PCA（Principal Component Analysis）的目标是找到数据中方差最大的方向，并用较少的主成分表示原始数据。它常用于降维、压缩和可视化。

我对 PCA 的理解是：它不是随意删掉一些特征，而是寻找最能保留数据变化信息的方向。通过把数据投影到这些方向上，可以在尽量少损失信息的情况下减少维度。

ICA（Independent Component Analysis）与 PCA 不同。PCA 强调不相关和方差最大方向，而 ICA 强调恢复相互独立的潜在成分。本周完成的鸡尾酒会实验正是 ICA 的典型应用。

如果多个麦克风同时接收到多个声源的线性混合，观测信号可以表示为：

```text
X = A S
```

其中 `S` 是原始独立声源，`A` 是混合矩阵。ICA 希望学习一个解混矩阵 `W`，使得：

```text
S ≈ X W^T
```

这部分让我意识到，无监督学习不仅能做聚类和降维，也能从混合信号中恢复隐藏的独立来源。

## 五、Lecture 16：MDP 与 Bellman Equations

Lecture 16 开始进入强化学习。MDP（Markov Decision Process）是强化学习中的基本数学框架，主要包括：

- `S`：状态集合；
- `A`：动作集合；
- `P(s' | s, a)`：状态转移概率；
- `R(s, a)`：奖励函数；
- `gamma`：折扣因子；
- `policy`：策略。

和监督学习不同，强化学习没有固定的正确标签。智能体需要根据当前状态选择动作，动作会影响后续状态和未来奖励。

Bellman Equation 的核心思想是：一个状态的价值等于当前奖励加上未来状态价值的折扣期望。它把“长期回报”拆成了当前一步和未来部分，使序列决策问题可以递归求解。

我对这一讲的理解是，强化学习最重要的变化在于目标从“单次预测正确”变成了“长期累计收益最大”。当前看起来收益高的动作，不一定是长期最优动作。

## 六、Lecture 17：Value Iteration、Policy Iteration、LQR 与 LQG

Lecture 17 继续讲强化学习与控制。Value Iteration 和 Policy Iteration 都是求解最优策略的方法。

Value Iteration 的思路是不断更新每个状态的价值估计，直到价值函数收敛，然后根据价值函数选择最优动作。

Policy Iteration 则分为两个步骤：

1. Policy Evaluation：在当前策略下计算价值函数；
2. Policy Improvement：根据价值函数改进策略。

这让我理解到，强化学习中的“学策略”不是凭空完成的，而是通过价值函数和策略之间的相互更新逐步逼近最优解。

LQR 和 LQG 属于控制问题中的重要内容，我目前只是初步理解：它们处理的是带有线性动态系统和二次代价函数的控制问题。相比离散状态下的 MDP，LQR/LQG 更偏连续控制和系统动力学，后续还需要继续补。

## 七、ICA 鸡尾酒会编程实验

### 1. 实验任务

本次实验来自 CS229 Problem Set 4 的 `p04_ica.py`。实验数据为 `mix.dat`，包含 5 路混合音频。任务是通过 ICA 学习一个解混矩阵 `W`，把混合音频分离成独立声源。

我在仓库中整理的实验路径为：

```text
week3task/task05_ica_cocktail_party/
```

核心代码文件为：

```text
week3task/task05_ica_cocktail_party/src/p04_ica.py
```

### 2. 核心实现

本次主要补全两个函数：

```python
def update_W(W, x, learning_rate):
    source_estimate = W.dot(x)
    gradient = np.linalg.inv(W.T) - np.outer(np.sign(source_estimate), x)
    return W + learning_rate * gradient
```

```python
def unmix(X, W):
    return X.dot(W.T)
```

`update_W` 基于 Laplace 分布假设，对 `W` 做随机梯度上升；`unmix` 使用学习到的 `W` 对混合信号矩阵做线性变换。

### 3. 实验结果

运行脚本后，程序会生成：

- `mixed_0.wav` 到 `mixed_4.wav`：原始混合音频；
- `split_0.wav` 到 `split_4.wav`：分离后的音频；
- `W.txt`：学习到的解混矩阵。

本地已运行通过，`X.shape = (53442, 5)`，说明数据包含 53442 个时间点和 5 路混合信号。最终脚本能够输出 5 条分离后的音频。

## 八、本周收获

本周最大的收获是把无监督学习和强化学习放进了更大的机器学习任务图景中。

前两周更多是在做监督学习和神经网络分类，核心是通过标签学习输入到输出的映射。本周学习的第 13-15 讲让我看到，在没有标签时，模型仍然可以发现数据内部结构、低维潜变量和独立成分。ICA 鸡尾酒会实验尤其直观：模型没有被告知每个声源是什么，但可以利用独立性假设把混合信号分离出来。

第 16-17 讲则让我理解到，强化学习处理的是动态决策问题。智能体不是只预测一个标签，而是在环境中连续行动，通过奖励信号学习长期更优的策略。

## 九、目前仍需加强的部分

1. EM 算法的完整推导还不熟，尤其是为什么 E-step 和 M-step 能不断提高似然。
2. Factor Analysis、PCA、ICA 之间的区别和联系还需要继续整理。
3. ICA 更新公式中的 log determinant 项和 Laplace 分布梯度还需要再推导一遍。
4. Bellman Equation、Value Iteration 和 Policy Iteration 的代码实现还不够熟。
5. LQR/LQG 目前只停留在初步概念层面，需要后续结合例子继续学习。

## 十、下周计划

1. 复习第 13-17 讲的公式推导，重点整理 EM、ICA 和 Bellman Equation；
2. 继续完成 CS229 后续 problem set，把理论和代码对应起来；
3. 对强化学习部分做一个小例子，尝试手写 Value Iteration 或 Policy Iteration；
4. 继续更新 GitHub 学习仓库，保证每周报告、README 和实验代码都能复现。
