# 第五周学习报告：CS231n Assignment 1 本地复现与 CIFAR-10 图像分类实验

## 一、本周学习概述

本周的核心任务是把 CS231n Assignment 1 在本地完整复现出来，并围绕 CIFAR-10 图像分类任务完成实验、调参、可视化和组会汇报材料整理。和前几周偏理论或小规模练习不同，这周更像一次完整的小型机器学习工程实践：从数据集准备开始，到模型代码验证、实验设计、超参数搜索、结果分析、图表生成和 PPT 汇报，形成了一条完整的学习闭环。

本周工作目录为：

```text
week5task/task07_cs231n_assignment1/
```

主要产出包括：

- CIFAR-10 数据集与 CS231n Assignment 1 本地运行环境；
- kNN、Softmax、TwoLayerNet 的三档规模复现实验；
- N10000、N20000、N49000 三组训练规模的结果对比；
- 结构化实验结果 JSON；
- 中文 Markdown 实验报告；
- 8 张实验可视化图；
- 14 页组会汇报 PPT；
- PPT 对应逐页讲稿。

## 二、实验目的

本周实验的目的不是单纯追求 CIFAR-10 上的最高准确率，而是理解图像分类系统的底层组成：

1. 如何加载图像数据并划分 train / validation / test；
2. kNN、Softmax 和 TwoLayerNet 分别代表怎样的建模思路；
3. 为什么原始像素空间的距离度量存在局限；
4. 线性分类器如何通过损失函数和梯度下降学习参数；
5. 非线性隐藏层为什么能提高分类效果；
6. 验证集如何用于超参数选择；
7. 测试集为什么只能用于最终评估；
8. 如何把实验结果整理成可复盘、可汇报的材料。

这个任务正好承接 CS229 中的监督学习、优化和正则化概念，同时过渡到 CS231n 中更强调工程实现和视觉任务的深度学习方法。

## 三、数据集与实验设置

本周使用的数据集是 CIFAR-10。它包含 10 个类别，每张图像大小为 32×32×3。类别包括 plane、car、bird、cat、deer、dog、frog、horse、ship 和 truck。

本地复现实验使用了三个训练规模：

| 规模 | 训练集 | 验证集 | 测试集 |
|---|---:|---:|---:|
| N10000 | 10,000 | 1,000 | 1,000 |
| N20000 | 20,000 | 1,000 | 1,000 |
| N49000 | 49,000 | 1,000 | 1,000 |

其中 N49000 是 CS231n Assignment 1 常用的训练集规模。测试集本次使用 1,000 张，是为了在本地快速复现实验和控制运行时间。完整 CIFAR-10 测试集为 10,000 张，后续可以继续扩展。

## 四、本周完成的模型实验

### 1. kNN

kNN 是最直观的分类器。它没有真正的参数训练，只是在训练阶段保存训练样本，预测时计算测试样本与训练样本之间的 L2 距离，再根据最近邻标签投票。

本周实验中，kNN 主要调节 `k` 值。结果显示，随着训练集规模扩大，kNN 的测试准确率有一定提升，但整体仍然有限：

| 规模 | 最佳 k | 验证准确率 | 测试准确率 |
|---|---:|---:|---:|
| N10000 | 10 | 0.315 | 0.288 |
| N20000 | 8 | 0.313 | 0.305 |
| N49000 | 5 | 0.326 | 0.348 |

kNN 的主要局限是：原始像素空间中的 L2 距离并不等价于图像语义相似度。同一类图像可能因为光照、背景、角度不同而像素距离很远；不同类图像也可能在像素空间中很接近。因此 kNN 适合作为 baseline，但不适合作为最终图像分类方案。

### 2. Softmax 线性分类器

Softmax 分类器将图像展平成 3,072 维向量，通过线性打分函数得到每个类别的 score，再使用 Softmax 和交叉熵损失训练参数。

本周主要调节 learning rate 和 regularization strength。在 N49000 上，搜索范围包括：

```text
learning_rate: 2.5e-7, 5e-7, 7.5e-7, 1e-6
regularization: 5e3, 1e4, 2.5e4, 5e4
```

结果如下：

| 规模 | 最佳学习率 | 最佳正则 | 验证准确率 | 测试准确率 |
|---|---:|---:|---:|---:|
| N10000 | 5e-7 | 1e4 | 0.349 | 0.359 |
| N20000 | 5e-7 | 1e4 | 0.375 | 0.355 |
| N49000 | 5e-7 | 5e3 | 0.383 | 0.359 |

Softmax 相比 kNN 更稳定，因为它能从数据中学习类别模板。但它仍然是线性模型，只能学习线性决策边界，因此在 CIFAR-10 这种复杂视觉任务上准确率上限明显。

### 3. TwoLayerNet

TwoLayerNet 在 Softmax 前加入一个隐藏层和 ReLU 激活函数。形式上可以写成：

```text
scores = W2 * ReLU(W1 * x + b1) + b2
```

这个模型的关键是非线性。没有 ReLU 的多层线性变换仍然等价于一个线性分类器；加入 ReLU 后，模型可以组合多个特征方向，表达更复杂的决策边界。

本周在 N49000 上尝试了 8 组代表性配置，调节 hidden_dim、learning_rate、reg、epochs 和 batch_size。最佳验证配置为：

```text
hidden_dim = 100
learning_rate = 2.5e-4
reg = 0.05
epochs = 8
batch_size = 512
```

主要结果如下：

| 规模 | 最佳配置概述 | 验证准确率 | 测试准确率 |
|---|---|---:|---:|
| N10000 | H=200, lr=5e-4, reg=0.25 | 0.466 | 0.455 |
| N20000 | H=100, lr=5e-4, reg=0.1 | 0.454 | 0.448 |
| N49000 | H=100, lr=2.5e-4, reg=0.05 | 0.515 | 0.493 |

TwoLayerNet 是本周表现最好的模型，说明即使只加入一层隐藏层，非线性表达能力也能显著提升原始像素上的分类效果。

## 五、结果对比与分析

三类模型在 N49000 上的测试准确率为：

| 模型 | 验证准确率 | 测试准确率 |
|---|---:|---:|
| kNN | 0.326 | 0.348 |
| Softmax | 0.383 | 0.359 |
| TwoLayerNet | 0.515 | 0.493 |

整体结论非常清楚：

```text
TwoLayerNet > Softmax > kNN
```

原因可以从模型表达能力解释：

- kNN 依赖原始像素距离，无法学习抽象特征；
- Softmax 能学习线性类别模板，但表达能力受线性边界限制；
- TwoLayerNet 通过隐藏层和 ReLU 学习非线性特征组合，因此泛化效果最好。

从数据规模看，kNN 的提升相对有限，说明它的瓶颈主要是距离度量，而不是数据量。Softmax 随数据规模增加有小幅提升。TwoLayerNet 在 N49000 上提升明显，说明可训练的非线性模型更能利用更多数据。

## 六、本周可视化与汇报材料

本周生成了以下可视化图表：

- 模型测试准确率对比；
- 模型验证准确率对比；
- Softmax 超参数热力图；
- TwoLayerNet 配置对比；
- 数据规模影响折线图；
- 运行耗时对比图；
- TwoLayerNet loss 曲线；
- 最佳模型混淆矩阵。

这些图表保存在：

```text
week5task/task07_cs231n_assignment1/results/figures/
```

组会汇报 PPT 为：

```text
week5task/task07_cs231n_assignment1/results/week5_cs231n_assignment1_report.pptx
```

PPT 对应讲稿为：

```text
week5task/task07_cs231n_assignment1/results/week5_cs231n_assignment1_speaker_notes.md
```

## 七、本周学习收获

这周最大的收获是把图像分类从概念推进到了可运行、可验证、可汇报的完整实验。

第一，我更清楚地理解了图像分类 pipeline：数据划分、模型训练、验证集调参、测试集评估是一个完整流程，不能只看单次训练结果。

第二，我理解了 kNN 的直观性和局限性。kNN 很适合作 baseline，但原始像素 L2 距离并不能很好表示图像语义。

第三，我理解了 Softmax 线性分类器的作用。它能学习参数，也能通过正则化控制过拟合，但线性边界限制了它在复杂视觉任务上的表现。

第四，我通过 TwoLayerNet 看到了非线性模型的优势。隐藏层和 ReLU 带来的表达能力，是神经网络优于线性模型的重要原因。

第五，我练习了调参和实验管理。不同 learning rate、regularization 和 hidden_dim 对结果影响明显，实验结果必须结构化保存，否则很难复盘。

## 八、当前局限与下一步计划

当前实验仍然有几个局限：

1. raw pixels 基线和 HOG + color histogram 手工特征实验均已完成；
2. TwoLayerNet 只做了代表性调参，还没有做非常细的网格搜索；
3. 测试集使用 1,000 张快速评估，后续可以扩展到完整 10,000 张；
4. 还没有进入 CNN，因此没有利用图像局部空间结构。

下一步计划：

1. 继续完成 `features.ipynb`，比较 HOG / color histogram 对 Softmax 的提升；
2. 继续完成 `FullyConnectedNets.ipynb`，学习多层网络、Dropout 和 Batch Normalization；
3. 增加 TwoLayerNet 的 epoch 数和更细调参；
4. 后续进入 CNN，理解卷积结构为什么更适合图像任务。

## 九、本周总结

Week5 通过 CS231n Assignment 1 的本地复现，把图像分类的基础方法系统串联起来。实验结果表明，原始像素空间下 kNN 表现有限，Softmax 能学习线性类别模板但上限不高，而 TwoLayerNet 通过非线性隐藏层显著提升了分类准确率。

这周的重点不是获得最高分，而是理解深度学习图像分类模型是如何从数据、损失函数、梯度优化、正则化和调参一步步搭建起来的。这为后续学习图像特征、FullyConnectedNet 和 CNN 打下了基础。

## 十、补充：Image Features 实验

在前一版 Week5 报告里，我把 HOG / color histogram 写成后续工作。现在这部分已经补跑完成，对应 CS231n Assignment 1 的 `features.ipynb`。

补充实验使用 49,000 张训练图像、1,000 张验证图像和 1,000 张测试图像。先提取 HOG 与 HSV color histogram，再分别训练 Softmax 和 TwoLayerNet。

| 输入表示 | 模型 | 验证准确率 | 测试准确率 |
|---|---|---:|---:|
| raw pixels | Softmax | 0.383 | 0.359 |
| HOG + color histogram | Softmax | 0.417 | 0.420 |
| raw pixels | TwoLayerNet | 0.515 | 0.493 |
| HOG + color histogram | TwoLayerNet | 0.605 | 0.576 |

这个结果说明，手工图像特征确实能显著改善分类效果。Softmax 虽然仍是线性模型，但输入换成 HOG + color histogram 后，测试准确率从 0.359 提升到 0.420。TwoLayerNet 进一步把测试准确率提升到 0.576。这说明 A1 的重点不只是“换模型”，也包括“换表示”：更好的图像表示会让简单分类器也变强。

因此，当前 Week5 的结论需要更新为：本周已经完成 raw pixels 基线、HOG + color histogram 特征实验，以及两类输入表示下的 Softmax / TwoLayerNet 对比。剩余未展开的主要是 FullyConnectedNet 的 dropout / batch normalization、更完整的 10,000 张测试集评估，以及后续 CNN。
