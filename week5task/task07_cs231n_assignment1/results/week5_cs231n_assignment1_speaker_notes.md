# Week5 CS231n Assignment 1 组会汇报讲稿

## Slide 1 标题页

大家好，这周我主要完成的是 CS231n Assignment 1 的本地复现实验。这个实验围绕 CIFAR-10 图像分类展开，重点比较了三类模型：kNN、Softmax 线性分类器和 TwoLayerNet 两层神经网络。

这次工作不是为了提交课程作业，而是为了把图像分类的基础 pipeline 在本地完整跑通：包括数据下载、代码验证、调参、结果记录、可视化和汇报材料整理。最终在 49,000 张训练样本规模下，TwoLayerNet 达到了 0.515 的验证集准确率和 0.493 的测试集准确率。

## Slide 2 任务背景

CS231n Assignment 1 的核心价值，是让我们从最简单的图像分类方法逐步过渡到可训练的神经网络。它不是一上来就用 CNN，而是先用 kNN 说明原始像素空间的限制，再用 Softmax 展示线性分类器如何通过损失函数和梯度下降学习参数，最后用 TwoLayerNet 引入隐藏层和非线性表达能力。

所以这一周的目标不是追求最高准确率，而是理解“图像分类系统到底由哪些部分组成”：数据划分、模型、损失函数、优化器、正则化、超参数搜索和最终测试集评估。

## Slide 3 数据集介绍

本实验使用 CIFAR-10。它一共有 10 个类别，包括飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。每张图像是 32×32 的彩色图片，也就是 3,072 维原始像素。

官方 Assignment 1 常用的数据划分是 49,000 张训练集、1,000 张验证集，以及测试集。本地复现实验为了比较数据规模影响，额外跑了 10,000、20,000 和 49,000 三档训练规模。验证集和测试集在本次快速复现中都固定为 1,000 张。

## Slide 4 实验流程

整个流程可以概括为五步。

第一步，加载 CIFAR-10 并做 train / validation / test 划分。第二步，分别训练 kNN、Softmax 和 TwoLayerNet。第三步，在验证集上做超参数选择，例如 kNN 的 k、Softmax 的 learning rate 和 regularization，以及 TwoLayerNet 的 hidden size、learning rate、reg 和 epoch 数。第四步，用测试集报告泛化性能。第五步，把结果保存成 JSON、Markdown、图表和 PPT，方便复盘与汇报。

这里特别重要的是验证集和测试集的分工：验证集用于选模型，测试集只用于最终评估。

## Slide 5 kNN

kNN 是最直观的模型。它几乎没有训练过程，只是把训练样本保存下来。预测时，模型计算测试图像与所有训练图像之间的距离，然后让最近的 k 个样本投票。

实验结果显示，kNN 在 N49000 下的最佳 k 是 5，验证准确率为 0.326，测试准确率为 0.348。这个结果不高，原因是 kNN 直接依赖原始像素空间的 L2 距离。但图像的语义相似性和像素距离并不稳定：同一类图片如果背景、光照、位置不同，像素距离可能很远；不同类图片也可能在像素上很接近。

另外，kNN 的推理成本很高。训练样本越多，预测时要比较的样本也越多，所以它在大数据规模下不适合反复调参。

## Slide 6 Softmax

Softmax 是一个线性分类器。它把图片展平成向量，然后通过一个权重矩阵为每个类别打分，最后用 Softmax 和交叉熵损失进行训练。

我在 N49000 上搜索了 learning rate 和 regularization strength。结果显示，较好的区域集中在 learning rate 约 5e-7 到 7.5e-7、regularization 约 5e3 附近。最佳 Softmax 的验证准确率是 0.383，测试准确率是 0.359。

Softmax 比 kNN 稳定，因为它不是直接比较像素距离，而是从数据中学习类别模板。但它的决策边界仍然是线性的，所以表达能力有限。

## Slide 7 TwoLayerNet

TwoLayerNet 在 Softmax 前面加入了一个隐藏层和 ReLU 激活函数。形式上可以写成：先通过第一层线性变换和 ReLU 得到隐藏表示，再通过第二层线性分类器输出类别分数。

这一步的关键是非线性。没有 ReLU 的多层线性变换仍然等价于一个线性模型；加入 ReLU 后，模型才能组合多个特征方向，表达更复杂的分类边界。

在 N49000 规模下，最佳验证配置是 hidden_dim=100、learning rate=2.5e-4、reg=0.05、8 epochs，验证准确率达到 0.515，测试准确率达到 0.493。这个结果明显优于 kNN 和 Softmax。

## Slide 8 模型整体对比

从测试准确率看，三个模型的排序非常清楚：TwoLayerNet 最高，Softmax 居中，kNN 最低。

在 N49000 下，kNN 测试准确率是 0.348，Softmax 是 0.359，TwoLayerNet 是 0.493。TwoLayerNet 的提升说明，在 raw pixels 上，只使用线性分类器是不够的；即使是简单的一层隐藏层，也能显著增强模型表达能力。

这个对比也解释了为什么深度学习要强调“表示学习”：模型不是只在原始输入上分类，而是先学习一个更适合分类的中间表示。

## Slide 9 数据规模影响

这页展示不同训练规模对测试准确率的影响。

kNN 从 10k 到 49k 有提升，但提升相对有限，因为瓶颈不是数据量本身，而是原始像素距离这个度量不够好。Softmax 随数据规模增加有一定提升，但上限仍受线性模型限制。TwoLayerNet 对更多数据最敏感，在 49k 规模上表现最好。

这说明更多数据通常更有利于可训练、可表达复杂规律的模型；但如果特征或模型本身能力不足，单纯加数据不会带来同等幅度的提升。

## Slide 10 损失曲线与混淆矩阵

损失曲线用于观察训练是否正常下降。这里重新训练了一次最佳 TwoLayerNet 配置，用来保存 loss history 和混淆矩阵。整体来看，loss 下降说明优化过程是有效的。

混淆矩阵则展示模型最容易混淆哪些类别。CIFAR-10 中一些视觉相近的类别，例如猫和狗、汽车和卡车，通常更容易被混淆。这个图的价值不只是看总体准确率，而是帮助我们发现模型错误的结构。

## Slide 11 最佳模型总结

本轮实验按照验证集选择模型，最佳模型是 N49000 规模下的 TwoLayerNet，配置为 hidden_dim=100、learning rate=2.5e-4、reg=0.05、8 epochs，验证准确率是 0.515，测试准确率是 0.493。

需要注意的是，个别配置在测试集上可能会出现更高的单次结果，例如某个配置测试准确率达到 0.510，但验证集略低。严格来说，模型选择应该依据验证集，而不是反复看测试集。这也是本实验中一个很重要的机器学习规范。

## Slide 12 关键学习收获

这周最大的收获是把图像分类 pipeline 跑完整了。

第一，理解了 train / val / test 的分工。第二，理解了 kNN、Softmax 和 TwoLayerNet 的模型差异。第三，练习了超参数搜索和结果记录。第四，直观看到非线性模型相对线性模型的优势。第五，也体会到 raw pixels 的局限：即使 TwoLayerNet 能显著提升，准确率仍然不到 CNN 的水平。

## Slide 13 当前局限

当前实验还有几个局限。

第一，主要使用 raw pixels，没有充分使用 HOG 和 color histogram 这类手工特征。第二，TwoLayerNet 只做了代表性调参，还不是非常细的搜索。第三，测试集为了快速复现实验只用了 1,000 张，而 CIFAR-10 完整测试集是 10,000 张。第四，还没有进入 CNN，所以模型还没有利用图像的局部空间结构。

这些局限也正好构成下一步学习路线。

## Slide 14 下一步计划

下一步我会沿着 CS231n Assignment 1 的后半部分继续推进。

第一，跑 `features.ipynb`，加入 HOG 和颜色直方图，观察手工特征对 Softmax 的提升。第二，跑 `FullyConnectedNets.ipynb`，比较多层网络、Dropout 和 Batch Normalization。第三，进一步增加 TwoLayerNet 的 epoch 和更细调参。第四，后续进入 CNN，用卷积结构利用图像的局部空间关系。

总结来说，Week5 的任务完成了从传统机器学习分类器到简单神经网络的过渡，为后续学习 CNN 打基础。

## Slide 15：补充实验 - HOG + Color Histogram

这一页要说明，Assignment 1 后半部分不只是用 raw pixels，还要求我们尝试 higher-level representations。我补跑了 HOG 和 HSV color histogram。HOG 主要描述边缘和梯度方向，color histogram 描述颜色分布。它们不是深度学习自动学出来的特征，而是传统计算机视觉里手工设计的图像表示。

这里最关键的对比是：Softmax 在 raw pixels 上测试准确率是 0.359，换成 HOG + color histogram 后提升到 0.420；TwoLayerNet 在 raw pixels 上是 0.493，换成特征后提升到 0.576。这说明输入表示本身非常重要。

## Slide 16：特征版调参结果

这一页展示两个调参图。左边是 Softmax 在特征上的 learning rate 和 regularization 搜索，右边是 TwoLayerNet 不同 hidden_dim、learning_rate 和 regularization 配置的比较。最佳 TwoLayerNet 特征版模型验证准确率达到 0.605，测试准确率达到 0.576。

这里可以强调，我们选择模型仍然依据 validation accuracy，而不是直接挑 test accuracy 最高的配置。这和前面 raw pixels 实验保持一致。

## Slide 17：更新后的结论

补充 features 实验之后，Week5 的结论更完整了。第一，raw pixels 可以跑通完整 pipeline，但表达能力有限。第二，HOG + color histogram 这样的手工特征能显著提升 Softmax 和 TwoLayerNet。第三，后续 CNN 的意义就更清楚了：CNN 不再依赖人手写 HOG，而是通过卷积层自动学习局部边缘、纹理和更高级的视觉模式。
