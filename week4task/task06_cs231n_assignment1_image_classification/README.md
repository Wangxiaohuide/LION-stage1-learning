# Task06: CS231n Assignment 1 图像分类与 CS229 后半段强化学习

## 任务背景

本周任务参考 Stanford CS231n 官方仓库 `cs231n/cs231n.github.io`，重点对应课程前五讲/Module 1 的内容：

| CS231n 内容 | 本周理解重点 |
|---|---|
| Image Classification | 数据驱动方法、train/val/test 划分、kNN |
| Linear Classification | 参数化分类器、SVM/Softmax、正则化 |
| Optimization | 损失函数、解析梯度、数值梯度、SGD |
| Backpropagation | 计算图、链式法则、反向传播 |
| Neural Networks Part 1 | 激活函数、两层网络、表达能力 |

同时，本周也补完 CS229 第 18 讲之后的内容，把重点放在 Q-learning、价值函数近似、Policy Search、REINFORCE、POMDPs 和课程总结上。

## 官方材料

已将 CS231n 2026 Assignment 1 starter code 放入：

```text
official_assignment1_starter/
```

该作业覆盖：

- `knn.ipynb`: k-Nearest Neighbor classifier
- `softmax.ipynb`: Softmax classifier
- `two_layer_net.ipynb`: Two-layer neural network
- `features.ipynb`: HOG/color histogram 等高层图像特征
- `FullyConnectedNets.ipynb`: fully connected network

## Assignment 1 内容与代码实现

本目录保留官方 Assignment 1 starter，并在 `official_assignment1_starter/cs231n/` 中补全了本周需要汇报的核心 Python 作业代码。Notebook 中的长时间 CIFAR-10 训练、超参数搜索结果和 inline question 仍需后续在 notebook 里继续运行和填写，但底层代码已经可以支持继续跑作业。

本次完成的代码文件包括：

| 文件 | 完成内容 |
|---|---|
| `cs231n/classifiers/k_nearest_neighbor.py` | 实现 kNN 的 two-loop、one-loop、no-loop L2 距离计算，以及最近邻标签投票预测 |
| `cs231n/classifiers/softmax.py` | 实现 Softmax naive / vectorized loss 与 gradient |
| `cs231n/classifiers/linear_classifier.py` | 实现 minibatch 采样、SGD 参数更新、预测函数，并补充线性 SVM vectorized loss |
| `cs231n/layers.py` | 实现 affine、ReLU、batchnorm、layernorm、dropout、卷积、池化、spatial batchnorm、spatial groupnorm、SVM loss、Softmax loss |
| `cs231n/classifiers/fc_net.py` | 实现 `TwoLayerNet` 和 `FullyConnectedNet` 的参数初始化、forward、loss、backward |
| `cs231n/optim.py` | 实现 SGD momentum、RMSProp、Adam 三种优化器 |

本次验证脚本位于：

```text
outputs/week4_assignment1_tests/verify_assignment1_core.py
```

验证覆盖 kNN 距离与预测、Softmax naive/vectorized 一致性、affine/ReLU/Softmax 层、TwoLayerNet 梯度、FullyConnectedNet forward/backward 与训练接口，并额外检查 batchnorm、layernorm、dropout、卷积、池化、spatial norm 和优化器的基本可运行性。

## 本任务是否是独立小任务

Task06 可以作为第四周的一个独立小任务汇报。它和前三周的 CS229 学习保持连续，但本周任务本身可以拆成三块：

1. 课程学习：学习 CS231n 前五讲和 CS229 第 18 讲之后内容；
2. 作业实现：补全 CS231n Assignment 1 的核心 Python 代码；
3. 汇报整理：更新 README、周报和 PPT，把课程内容、代码实现和运行验证结果串起来。

## 本周收获

CS231n 的前几讲和 Assignment 1 把图像分类问题拆成了一条非常清楚的工程路径：先定义数据划分，再从 kNN 这种无训练参数的方法出发，过渡到线性分类器，最后进入可反向传播训练的神经网络。本周不只是整理 starter，而是把核心 Python 实现补上，并通过小规模测试确认主要接口能运行。

CS229 第 18 讲之后则把第三周已经开始的 MDP/Bellman 思路继续推进到无模型强化学习和策略优化。两门课在这里形成了一个很好的连接：CS231n 侧重“如何从图像中学表示并分类”，CS229 后半段侧重“如何在环境反馈中学习长期决策”。
