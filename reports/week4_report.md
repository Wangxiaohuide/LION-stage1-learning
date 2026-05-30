# 第四周学习报告：CS231n 前五讲、Assignment 1 与 CS229 后半段

## 一、本周学习概况

本周主要完成两部分内容：

1. 参考 `cs231n/cs231n.github.io` 官方仓库，学习 CS231n 前五讲/Module 1，并整理 Assignment 1 的图像分类主线；
2. 补完 CS229 第 18 讲之后的强化学习内容，包括 Q-learning、价值函数近似、Policy Search、REINFORCE、POMDPs 和课程总结。

本周新增任务目录为：

```text
week4task/task06_cs231n_assignment1_image_classification/
```

## 二、CS231n 前五讲理解

### 1. Image Classification

CS231n 从图像分类开始，把计算机视觉问题转化为一个数据驱动的学习流程：输入是一张图片，输出是固定类别集合中的一个标签。与手写规则不同，数据驱动方法通过训练集学习模型参数，再在验证集和测试集上评估泛化能力。

这一讲中最重要的思想是 train/validation/test split。训练集用于拟合参数，验证集用于选择超参数，测试集只用于最终评估。kNN 虽然实际效率不高，但它非常适合作为第一种分类器：训练阶段只是记住数据，预测阶段再根据距离寻找最近样本。

### 2. Linear Classification

线性分类器把图像拉平成向量后，用

```text
scores = W x + b
```

为每个类别打分。SVM 和 Softmax 的差别不在模型形式，而在损失函数。SVM 强调正确类别分数要比错误类别高出 margin；Softmax 把分数转成概率分布，并用交叉熵惩罚错误预测。

这部分让我更清楚地看到，模型结构、损失函数、正则化三者是分开的：同样的线性打分函数，可以配不同的损失；同样的损失，也可以加不同强度的 L2 regularization。

### 3. Optimization

优化部分回答了“如何找到好的参数”。损失函数给出了评价标准，但真正训练模型需要计算梯度并更新参数。数值梯度适合理解和检查，解析梯度适合真正训练。

SGD 的核心是用小批量数据估计整体梯度，从而在计算成本和更新频率之间取得平衡。学习率是最敏感的超参数之一：太大容易震荡，太小训练很慢。

### 4. Backpropagation

反向传播本质上是链式法则在计算图上的系统应用。每个局部节点只需要知道自己的输入、输出和上游梯度，就能把梯度继续传回前面的参数。

这部分和 CS229 的梯度推导形成呼应：CS229 更偏公式推导，CS231n 更强调把复杂表达式拆成计算图，再按模块实现 forward/backward。

### 5. Neural Networks Part 1

两层神经网络可以写成：

```text
scores = W2 ReLU(W1 x + b1) + b2
```

如果没有 ReLU 等非线性，多层线性变换仍然等价于一个线性分类器。非线性激活函数是神经网络获得更强表达能力的关键。

## 三、CS231n Assignment 1 整理

官方 Assignment 1 starter 已经放在：

```text
week4task/task06_cs231n_assignment1_image_classification/official_assignment1_starter/
```

本次作业的目标包括：

- 理解图像分类 pipeline；
- 实现 kNN；
- 实现 Softmax classifier；
- 实现 two-layer neural network；
- 使用 HOG、color histogram 等高层图像特征；
- 训练 fully connected network。

当前仓库保留官方 starter code 和 notebook，用于后续逐题补全 Assignment 1。第四周不包含额外自写脚本，避免和官方作业材料混在一起。

## 四、CS229 第 18 讲之后内容

### Lecture 18: Q-Learning and Value Function Approximation

Q-learning 解决的是不知道环境转移概率时如何学习最优动作价值函数的问题。它通过与环境交互，不断更新：

```text
Q(s, a) <- Q(s, a) + alpha [r + gamma max_a' Q(s', a') - Q(s, a)]
```

这可以理解为用当前估计和一步后看到的回报之间的 TD error 来修正 Q 值。Value function approximation 则进一步解决状态空间太大时无法保存完整表格的问题，用参数化函数近似价值函数。

### Lecture 19: Policy Search, REINFORCE, POMDPs

Policy Search 不再先学价值函数，而是直接优化策略参数。REINFORCE 使用采样轨迹估计策略梯度，让高回报动作在未来更可能被选择。

POMDP 则处理状态不能被完全观测的情况。智能体看到的是 observation，而不是真实 state，因此需要在不确定性下维护 belief 或借助历史信息做决策。

### Lecture 20: Optional Topics and Wrap-up

最后一讲更像课程总结，把监督学习、无监督学习、强化学习和应用专题串起来。对我来说，CS229 后半段的主线是：从已知模型的 Bellman equation，到未知模型下的 Q-learning，再到直接优化策略的 policy gradient。

## 五、本周收获

本周最大的收获是把“图像分类”和“序列决策”两条线放在一起理解。

CS231n 前五讲让我看到，从原始像素到分类结果，中间需要经历数据划分、特征表示、损失函数、梯度优化和反向传播。Assignment 1 则把这些概念变成了非常具体的代码任务。

CS229 第 18 讲之后让我继续理解强化学习：当没有固定标签、只有环境反馈时，模型要学的不再是一次预测，而是长期回报最大的行动策略。Q-learning 和 Policy Search 分别代表了基于价值函数和直接优化策略的两种路线。

## 六、后续需要加强

1. 回到 CS231n 官方 notebook，逐个补全 Assignment 1 的 TODO；
2. 对 Softmax 梯度和两层网络反向传播再做一次手推；
3. 使用 CIFAR-10 数据完成官方 kNN、Softmax、two-layer net、features 和 fully connected network notebook；
4. 对比 tabular Q-learning、value function approximation 和 policy gradient 的适用场景；
5. 继续保持每周 README、报告、代码和运行记录同步更新到 GitHub。
