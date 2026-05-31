# 第四周学习报告：CS231n 前五讲、Assignment 1 与 CS229 后半段

## 一、本周学习概况

本周的学习主线分成两条：一条是 CS231n 前五讲对应的图像分类基础，另一条是 CS229 第 18 讲之后的强化学习内容。和前三周相比，这一周的重点更偏“把课程概念落到一个真实作业框架里”：CS231n Assignment 1 不是单独的一道小题，而是把图像分类从数据处理、kNN、线性分类器、Softmax、两层神经网络一直串到图像特征和全连接网络。

本周新增任务目录为：

```text
week4task/task06_cs231n_assignment1_image_classification/
```

其中官方 Assignment 1 starter 位于：

```text
week4task/task06_cs231n_assignment1_image_classification/official_assignment1_starter/
```

本周完成的工作主要包括：

1. 参考 `cs231n/cs231n.github.io` 官方课程与作业材料，整理 CS231n 前五讲的知识结构；
2. 下载并归档 CS231n Assignment 1 starter，梳理 notebook 和 `cs231n/` 代码框架中的 TODO；
3. 对 Assignment 1 的 kNN、Softmax、linear classifier 三个基础代码作业进行实现，并阅读 two-layer net、features、FullyConnectedNets 的后续实验结构；
4. 补充学习 CS229 第 18 讲之后的 Q-learning、value function approximation、policy search、REINFORCE、POMDPs 和课程总结；
5. 对比 CS231n 图像分类任务和 CS229 强化学习任务的差别：前者偏“给定数据集上的监督分类”，后者偏“和环境交互下的长期决策”。

需要说明的是：本周 Assignment 1 分两步推进。前半段先完成官方 starter 的归档、阅读、结构拆解和初步运行检查；随后重点完成三个基础代码作业：`k_nearest_neighbor.py`、`softmax.py`、`linear_classifier.py`。Notebook 中的 CIFAR-10 完整运行、模型准确率对比、训练曲线和 inline question 计划放到下周继续完善。

## 二、CS231n 前五讲学习内容

### 1. Image Classification：从规则系统到数据驱动

CS231n 第一部分从图像分类开始。图像分类看起来是一个简单问题：输入一张图片，输出它属于哪个类别。但真正困难的是，图片中的物体会受到视角、光照、遮挡、背景、形变等因素影响，很难靠手写规则解决。

这部分让我重新理解了“数据驱动方法”的意义。传统规则系统试图人工写出识别规则，而机器学习方法是先准备训练数据，再让模型从数据中学习分类规律。图像分类 pipeline 可以概括为：

```text
收集数据 -> 划分 train/val/test -> 选择模型 -> 训练参数 -> 验证集调参 -> 测试集评估
```

其中 train/validation/test split 是我觉得最重要的一点。训练集用于学习参数，验证集用于选择超参数，测试集只用于最终评估。如果把测试集提前用于调参，就会把模型“调”到测试集上，最后得到的准确率不再能代表真实泛化能力。

kNN 是这一讲中最直观的分类器。它的训练阶段几乎没有计算，只是把训练样本保存下来；预测阶段再计算测试图片和训练图片之间的距离，并根据最近的样本投票。它的优点是容易理解，缺点也很明显：预测阶段计算量大，对距离度量敏感，并且没有真正学到抽象特征。

我的理解是，kNN 在 CS231n 中更像一个“起点实验”：它让我们看到图像分类可以被转化为距离比较，但也暴露出直接在像素空间做比较的局限。两张语义相同的图片，只要位置、背景或亮度稍微变化，像素距离就可能很大。

### 2. Linear Classification：把图像分类写成参数化模型

线性分类器把图像拉平成一个向量，然后通过参数矩阵和偏置为每个类别打分：

```text
scores = W x + b
```

这部分对我很重要，因为它把“分类器”从 kNN 的记忆式方法推进到了参数化模型。模型不再保存所有训练样本，而是学习一组参数 `W` 和 `b`。这些参数可以看成每个类别的模板：当输入图像和某个类别模板更匹配时，该类别得分更高。

SVM 和 Softmax 的区别主要在损失函数：

- SVM loss 强调正确类别分数要比错误类别至少高出一个 margin；
- Softmax loss 把分数转成概率分布，再用交叉熵惩罚错误分类。

我自己的理解是，SVM 更像是在问“正确类别有没有比其他类别明显更好”，Softmax 更像是在问“模型给正确类别分配了多大的概率”。这两种损失函数都使用同一个线性打分模型，但优化目标不同，因此梯度形式也不同。

这一讲也让我更清楚地区分了三个概念：

1. 模型结构：例如 `scores = W x + b`；
2. 损失函数：例如 SVM loss 或 Softmax cross-entropy；
3. 正则化：例如 L2 regularization，用来抑制参数过大、减轻过拟合。

之前容易把这些东西混在一起看，现在更能理解：同一个模型结构可以配不同损失，同一个损失也可以加不同正则化强度。

### 3. Optimization：从损失函数走向参数更新

优化部分解决的问题是：有了损失函数以后，如何找到能让损失变小的参数。

课程中先讲数值梯度，再讲解析梯度。数值梯度通过微小扰动参数来观察 loss 的变化，适合理解和检查；解析梯度通过公式推导或反向传播直接计算，适合真正训练模型。Assignment 1 中很多实验都要求同时比较 naive 实现和 vectorized 实现，本质上也是在训练我理解“正确性”和“效率”的区别。

SGD 的思想是每次只用一小批样本估计梯度，而不是每次都遍历完整训练集。这样虽然每一步方向有噪声，但更新更频繁、计算成本更低。学习率是这里最敏感的超参数：学习率太大可能导致 loss 震荡甚至发散；学习率太小又会让训练非常慢。

我对这一讲的思考是：优化不是一个“公式代入”过程，而是一个工程问题。即使模型和损失函数都写对了，如果学习率、正则化强度、batch size 或初始化不合适，训练结果仍然会很差。这也是为什么 Assignment 1 里会安排交叉验证和超参数搜索。

### 4. Backpropagation：把复杂梯度拆成局部计算

反向传播是 CS231n 前几讲中最关键的内容之一。它的本质是链式法则在计算图上的系统应用。一个复杂函数可以拆成很多局部节点，每个节点只需要知道自己的输入、输出和上游梯度，就能计算下游变量对自己输入的梯度。

这部分和 CS229 里偏数学推导的梯度计算形成了互补。CS229 更强调从公式上推导目标函数的梯度；CS231n 更强调把神经网络写成一层层 forward/backward 模块。对于写代码来说，CS231n 的方式更接近真实工程实现。

我目前的理解是：反向传播并不是神经网络特有的“神秘算法”，而是把链式法则组织成一种高效的计算方式。真正需要训练的是把每个模块的 forward 输出保存好，然后 backward 时按相反顺序传回梯度。

### 5. Neural Networks Part 1：两层网络与非线性表达能力

两层神经网络可以写成：

```text
scores = W2 ReLU(W1 x + b1) + b2
```

如果没有 ReLU 这样的非线性激活函数，多层线性变换仍然等价于一个线性分类器。因此神经网络的表达能力并不是来自“层数本身”，而是来自线性变换和非线性激活的组合。

这部分让我理解到，神经网络可以看成一种自动学习特征的方式。线性分类器直接在原始像素上分类，而两层网络先通过隐藏层把输入变换到新的表示空间，再进行分类。隐藏层学习到的表示如果更适合区分类别，最终分类效果就会更好。

Assignment 1 中的 `two_layer_net.ipynb` 正好对应这部分内容：需要手动实现 forward pass、loss、backward pass 和参数更新。这个实验比 kNN 和 Softmax 更综合，因为它同时涉及矩阵维度、ReLU、Softmax loss、L2 正则化和反向传播。

## 三、CS231n Assignment 1 实验内容整理

### 1. 官方 starter 结构

本周归档的官方 starter 包含以下主要 notebook：

| 文件 | 实验主题 | 需要完成的核心内容 |
|---|---|---|
| `knn.ipynb` | k-Nearest Neighbor classifier | 距离计算、标签投票、交叉验证选择 `k` |
| `softmax.ipynb` | Softmax classifier | naive/vectorized loss 和 gradient |
| `two_layer_net.ipynb` | Two-layer neural network | 初始化、前向传播、反向传播、训练和预测 |
| `features.ipynb` | Image features | HOG、color histogram 特征与分类器训练 |
| `FullyConnectedNets.ipynb` | Fully connected networks | 多层全连接网络、优化器、正则化等 |

对应的核心代码目录为：

```text
official_assignment1_starter/cs231n/
```

其中和本周阅读最相关的文件包括：

```text
cs231n/classifiers/k_nearest_neighbor.py
cs231n/classifiers/softmax.py
cs231n/classifiers/fc_net.py
cs231n/layers.py
cs231n/solver.py
cs231n/optim.py
```

### 2. kNN 实验理解

kNN 实验要求实现三种距离计算方式：

1. two loops：两层循环，逐个测试样本和训练样本计算距离；
2. one loop：只循环测试样本，对所有训练样本做向量化计算；
3. no loops：完全用矩阵运算和广播计算距离矩阵。

这三个版本的结果应该一致，但运行速度不同。这个实验的意义不只是实现 kNN，更重要的是训练 NumPy 向量化思维。尤其是 no-loop 版本，需要把欧式距离写成：

```text
||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y
```

这样就能通过矩阵乘法一次性得到所有测试样本和训练样本之间的距离。

我自己的思考是：这部分是从“数学公式”到“高效代码”的典型转换。公式本身不难，但要写成不显式循环的矩阵代码，需要非常清楚每个数组的 shape。CS231n 的作业很强调这一点，因为后面神经网络的 forward/backward 也都是 shape 驱动的。

### 3. Softmax 实验理解

Softmax 实验要求实现 naive 和 vectorized 两个版本。naive 版本用循环逐个样本计算 loss 和 gradient；vectorized 版本则一次性计算整个 batch 的分数、概率、loss 和梯度。

Softmax 的核心步骤是：

```text
scores = X W
scores -= max(scores)
probabilities = exp(scores) / sum(exp(scores))
loss = -log(probability of correct class)
```

减去 `max(scores)` 是为了数值稳定，避免指数运算溢出。这个细节让我意识到，机器学习代码不是只要公式对就行，还要考虑浮点数计算的稳定性。

Softmax 梯度部分是我后续需要重点补的内容。它看起来比 SVM loss 更“概率化”，但实现时可以理解为：先得到预测概率矩阵，再把正确类别位置减 1，最后乘上输入矩阵得到 `dW`。这个推导需要再手写一遍，才能真正写稳。

### 4. Two-layer network 实验理解

两层网络实验把前面几部分串起来。它不再只是一个线性分类器，而是：

```text
input -> affine -> ReLU -> affine -> Softmax
```

需要实现的内容包括：

- 参数初始化：`W1, b1, W2, b2`；
- 前向传播：计算 hidden layer 和 scores；
- 损失函数：Softmax loss 加 L2 regularization；
- 反向传播：计算每个参数的梯度；
- 训练循环：使用 SGD 更新参数；
- 预测函数：根据 scores 选择类别。

我觉得这个实验的价值在于，它强迫我把“反向传播”写成具体数组操作。只看公式时容易觉得理解了，但真正写代码时会遇到很多细节，例如 ReLU backward 要根据 hidden layer 是否大于 0 来挡住梯度，正则化项的梯度要加到对应的 `dW` 上，bias 的梯度要按 batch 维度求和。

### 5. Features 与 FullyConnectedNets 实验理解

`features.ipynb` 把重点从原始像素转向手工特征。HOG 特征关注局部梯度方向，color histogram 关注颜色分布。这个实验让我看到，在深度学习完全端到端之前，特征工程仍然是视觉任务中非常重要的一环。

`FullyConnectedNets.ipynb` 则把两层网络推广到任意层数的全连接网络，并引入更完整的训练框架。它和 `fc_net.py`、`layers.py`、`solver.py`、`optim.py` 联系紧密，后续需要补全 affine、ReLU、Softmax、优化器等模块。

这部分本周已经从“结构阅读”推进到“基础代码实现”。也就是说，notebook 里的 CIFAR-10 完整训练和结果记录还没有全部完成，但 kNN、Softmax 和 linear classifier 三个基础模块已经先补上，后续可以继续按 notebook 顺序跑 CIFAR-10 实验和调参。

### 6. 当前实验状态

本周当前状态可以概括为：

- 已完成：官方 starter code 和 notebook 的归档；
- 已完成：Assignment 1 的实验结构、核心文件和 TODO 范围梳理；
- 已完成：kNN、Softmax、linear classifier 的任务目标理解和代码实现；
- 已阅读：two-layer net、features、FullyConnectedNets 的后续实验结构；
- 已完成：`k_nearest_neighbor.py` 中三种距离计算和标签投票；
- 已完成：`softmax.py` 中 naive / vectorized Softmax loss 与 gradient；
- 已完成：`linear_classifier.py` 中 minibatch 采样、SGD 更新、预测函数和 SVM vectorized loss；
- 已完成：用小规模验证脚本检查核心实现可运行；
- 下周计划：在 CIFAR-10 上运行 Assignment 1 实验，记录各模型验证集/测试集准确率对比；
- 下周计划：继续完成 Stanford CS231n 后续课程学习。

这部分要如实记录：第四周的 Assignment 1 已经完成三个基础代码作业，但还不是“完整提交版作业”。完整提交版还需要下周继续跑 CIFAR-10 notebook，补充训练曲线、验证集/测试集准确率结果和文字回答。

## 四、CS229 第 18 讲之后学习内容

### 1. Q-learning

CS229 第 18 讲之后继续强化学习。Q-learning 解决的是不知道环境转移概率时，如何通过交互学习动作价值函数的问题。它的更新公式为：

```text
Q(s, a) <- Q(s, a) + alpha [r + gamma max_a' Q(s', a') - Q(s, a)]
```

这里括号中的部分是 TD error，表示当前估计和“实际看到的一步奖励加未来最优价值”之间的差距。Q-learning 的特点是 off-policy：它学习的是最优策略对应的价值，即使采样行为本身可能带有探索。

我的理解是，Q-learning 和前面监督学习最大的差别在于没有固定标签。监督学习中每个样本通常有一个标准答案，而 Q-learning 的“答案”来自环境反馈和未来价值估计，是边探索边修正的。

### 2. Value Function Approximation

表格型 Q-learning 只适合状态和动作空间比较小的情况。如果状态空间很大，无法为每个 `(s, a)` 都保存一个表格值，就需要用函数近似：

```text
Q(s, a) ≈ f_theta(s, a)
```

这和神经网络有很强的联系。CS231n 中神经网络学习的是从图像到类别分数的函数；强化学习中的 value function approximation 学习的是从状态动作到长期回报的函数。两者都在做函数拟合，但监督信号的来源不同。

我觉得这正好把 CS231n 和 CS229 联系起来：图像分类里网络输出 class score，强化学习里网络可以输出 value 或 Q-value。前者优化分类损失，后者优化 Bellman error 或策略目标。

### 3. Policy Search 与 REINFORCE

Policy Search 不再先学习价值函数，而是直接优化策略：

```text
pi_theta(a | s)
```

REINFORCE 用采样轨迹估计策略梯度。直观理解是：如果某条轨迹获得了高回报，那么这条轨迹中出现过的动作在相应状态下应该更容易被选择；如果回报低，则应该降低这些动作的概率。

这部分让我意识到，强化学习可以有两条路线：

1. value-based：先学价值函数，再根据价值选动作；
2. policy-based：直接学策略，让策略本身朝高回报方向更新。

Q-learning 属于第一类，REINFORCE 属于第二类。它们的共同点是都要处理延迟奖励和探索问题，区别在于优化对象不同。

### 4. POMDPs

POMDP 处理的是状态不能被完全观测的情况。MDP 假设智能体能看到完整状态，但现实中很多任务只能看到 observation。例如自动驾驶不一定能知道所有车辆的真实意图，机器人也可能只能通过传感器获得部分信息。

POMDP 中需要根据历史观测维护 belief state。我的理解是，这让强化学习从“当前状态决策”变成了“在不完整信息下根据历史推断当前情况再决策”。这比标准 MDP 更接近真实世界，也更难。

### 5. 课程总结

CS229 后半段把机器学习从监督学习扩展到无监督学习、强化学习和控制。对我来说，这一段最重要的收获是理解不同任务的监督信号来自哪里：

- 监督学习：标签提供直接目标；
- 无监督学习：数据结构本身提供学习信号；
- 强化学习：环境奖励提供延迟反馈；
- 控制问题：系统动态和代价函数共同定义优化目标。

这让我重新看 CS231n 的图像分类：它是监督学习中非常典型、也非常工程化的一类问题。而 CS229 后半段则提醒我，机器学习不只是分类和回归，还包括在环境中连续做决策。

## 五、本周自己的思考

本周最大的思考是：课程内容和编程作业之间不能只停留在“看懂概念”，必须能对应到代码位置。

例如，CS231n 讲 kNN 时，概念上只是“找最近邻投票”，但 Assignment 1 中真正要写的是三种距离计算、标签排序、投票和交叉验证。CS231n 讲 Softmax 时，公式上只是概率和交叉熵，但代码中要考虑数值稳定、正则化、矩阵维度和梯度向量化。CS231n 讲反向传播时，理论上是链式法则，但在 `fc_net.py` 和 `layers.py` 里就变成了每一层 forward cache 和 backward gradient 的组织问题。

我也发现自己目前的薄弱点比较清楚：

1. 对 NumPy shape 的敏感度还不够，尤其是广播和矩阵乘法；
2. Softmax gradient 虽然知道大方向，但还需要手推和代码实现结合；
3. 神经网络 backward pass 不能只背公式，要能按模块拆开；
4. 强化学习部分理解了概念，但还缺少代码实验来巩固。

CS231n 和 CS229 放在一起学习的好处是，它们从不同角度训练同一个能力：把数学目标翻译成可运行代码。CS231n 更偏视觉和神经网络工程，CS229 更偏机器学习理论框架。两者结合起来，能帮助我既不只会调包，也不只停留在公式。

## 六、后续计划

下一步计划按 Assignment 1 的顺序继续推进，不跳过基础实验：

1. 下载并准备 CIFAR-10 数据集；
2. 在 `knn.ipynb` 中运行 kNN 实验，记录不同 `k` 下的验证集准确率；
3. 在 `softmax.ipynb` 中运行 Softmax 实验，记录训练集、验证集和测试集准确率；
4. 对比 kNN、Softmax、linear classifier 在 CIFAR-10 上的表现差异；
5. 继续完成 Stanford CS231n 后续课程内容；
6. 继续推进 Assignment 1 后面的 two-layer net、features 和 FullyConnectedNets 实验。

本周报告补充后的目标是：不仅记录“看了哪些课”，也记录“这些课和 Assignment 1 的代码任务如何对应”，并且如实保留当前实验进度，方便后续接着完成。

## 七、组会汇报口径

如果开组会汇报，本周 Task06 可以作为一个独立小任务来讲。它和前三周的 CS229 学习有连续性，但任务边界是清楚的：本周从 CS231n 图像分类入手，把课程前五讲对应到 Assignment 1 的代码实现，同时补完 CS229 后半段强化学习内容。

### 1. 这个任务整体做了什么

本周任务整体做了三件事：

1. 学习课程内容：CS231n 前五讲，包括图像分类、kNN、线性分类器、优化、反向传播和两层神经网络；同时学习 CS229 第 18 讲之后的 Q-learning、value function approximation、policy search、REINFORCE 和 POMDPs。
2. 完成代码实现：在 CS231n Assignment 1 starter 的基础上，先补全 `k_nearest_neighbor.py`、`softmax.py`、`linear_classifier.py` 三个基础代码作业。
3. 整理汇报材料：更新 README、周报和 PPT，把“学习了什么、实现了什么、验证了什么、下一步还要做什么”串起来。

### 2. 它是不是独立小任务

它可以看成第四周的独立小任务，任务名称可以叫：

```text
Task06：CS231n Assignment 1 基础分类器实现与 CS229 强化学习补完
```

独立性体现在：它有单独目录 `week4task/task06_cs231n_assignment1_image_classification/`，有独立的官方 starter code，有明确的三个代码实现文件，也有单独的周报和 PPT。和前三周的关系是学习主线连续，但本周交付物是新的。

### 3. 分别完成了什么

课程学习部分完成：

- CS231n Lecture 1：理解图像分类 pipeline、数据驱动方法和 train/val/test 划分；
- CS231n Lecture 2：理解线性分类器、SVM loss、Softmax loss 和 L2 正则化；
- CS231n Lecture 3：理解优化、数值梯度、解析梯度、SGD 和学习率影响；
- CS231n Lecture 4：理解反向传播和计算图；
- CS231n Lecture 5：理解两层神经网络、ReLU 和非线性表达能力；
- CS229 后半段：理解 Q-learning、价值函数近似、策略搜索、REINFORCE、POMDPs 和课程总结。

代码实现部分完成：

- `k_nearest_neighbor.py`：实现 two-loop、one-loop、no-loop 三种 L2 距离计算，实现最近邻投票预测；
- `softmax.py`：实现 naive 和 vectorized Softmax loss / gradient；
- `linear_classifier.py`：实现 minibatch 采样、SGD 更新、预测函数，并补充 SVM vectorized loss。

验证部分完成：

- 编写 `outputs/week4_assignment1_tests/verify_assignment1_core.py` 做小规模功能验证；
- 验证 kNN 距离和标签预测；
- 验证 Softmax naive/vectorized 结果一致；
- 验证 linear classifier 可以完成 minibatch 训练和预测；
- 使用 `compileall` 检查相关 Python 文件可以正常编译。

### 4. 还没有完成什么

需要在组会上如实说明：本周已经完成 Assignment 1 的三个基础代码作业，但还没有完成完整 notebook 级别的最终提交。下周还需要：

- 下载或准备 CIFAR-10 数据；
- 先按 `knn.ipynb`、`softmax.ipynb` 顺序运行基础分类实验；
- 记录各模型验证集/测试集准确率对比；
- 补齐 notebook 中的 inline question；
- 继续完成 Stanford CS231n 后续课程，并推进 two-layer net、features、FullyConnectedNets 实验。
