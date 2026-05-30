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

## Assignment 1 内容

本目录保留官方 Assignment 1 starter，作为后续逐题完成 notebook 的基础材料。当前第四周完成的是课程内容整理和作业结构归档，不包含额外自写脚本。

后续需要在官方 notebook 和 `cs231n/` 代码框架中继续补全：

- kNN 的距离计算、标签投票和交叉验证；
- Softmax loss 的 naive/vectorized 实现；
- two-layer neural network 的 forward/backward；
- 图像特征实验；
- fully connected network 的训练与调参。

## 本周收获

CS231n 的前几讲和 Assignment 1 把图像分类问题拆成了一条非常清楚的工程路径：先定义数据划分，再从 kNN 这种无训练参数的方法出发，过渡到线性分类器，最后进入可反向传播训练的神经网络。

CS229 第 18 讲之后则把第三周已经开始的 MDP/Bellman 思路继续推进到无模型强化学习和策略优化。两门课在这里形成了一个很好的连接：CS231n 侧重“如何从图像中学表示并分类”，CS229 后半段侧重“如何在环境反馈中学习长期决策”。
