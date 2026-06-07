# Week5 CS231n Assignment 1 PPT 逐页讲稿

## Slide 1：标题页
今天汇报 Week5 的 CS231n Assignment 1 本地复现实验。重点不是提交官方作业，而是在本地复现 CIFAR-10 图像分类流程。最终结果是：raw pixels 上最佳 TwoLayerNet 测试准确率为 0.493；加入 HOG + color histogram 后，最佳 TwoLayerNet 测试准确率提升到 0.576。

## Slide 2：Assignment 1 范围
Assignment 1 分成五块：kNN、Softmax、TwoLayerNet、Image Features 和 FullyConnectedNet。本次已经补齐 Q4，也就是 higher-level image features。Q5 的基础验证已经跑过，但 dropout 和 batch normalization 更适合放到 Assignment 2 继续展开。

## Slide 3：数据集与划分
实验使用 CIFAR-10，每张图片是 32×32×3。官方训练集 50,000 张，本次按 A1 常用方式划出 49,000 张训练和 1,000 张验证。测试为了本地快速评估，使用 1,000 张。

## Slide 4：实验流程
整个 pipeline 是先准备数据，再训练模型，再用验证集选择超参数，最后用测试集报告泛化性能。这里强调不能用测试集调参，否则测试结果就不再客观。

## Slide 5：Raw pixels 基线
raw pixels 基线中，kNN 测试准确率 0.348，Softmax 是 0.359，TwoLayerNet 是 0.493。结果说明非线性隐藏层确实比线性模型更强。

## Slide 6：Raw pixels 的局限
raw pixels 的问题是图像空间结构被打散。一个小的平移、背景变化或者亮度变化，都会让像素距离发生很大变化。所以 kNN 和 Softmax 在 raw pixels 上都有明显上限。

## Slide 7：Image Features 补充实验
这一页是这次新增重点。HOG 统计边缘和梯度方向，color histogram 统计颜色分布。它们让模型看到的不是零散像素，而是更接近视觉语义的手工特征。

## Slide 8：Raw pixels vs Image Features
表格显示，Softmax 从 0.359 提升到 0.420，TwoLayerNet 从 0.493 提升到 0.576。这说明表示学习或特征工程对图像分类非常重要。

## Slide 9：Softmax on Features
左边热力图展示 Softmax 在特征上的调参。最佳配置是 lr=5.0e-07, reg=5000，测试准确率 0.420。线性模型本身没变，但输入表示变好后结果明显提升。

## Slide 10：TwoLayerNet on Features
TwoLayerNet 在特征上继续提升。最佳配置是 hidden_dim=750, lr=1.0e-03, reg=0.001，验证准确率 0.605，测试准确率 0.576。

## Slide 11：实验解释
这一页解释为什么 features 有用。HOG 把边缘和轮廓编码出来，颜色直方图保留颜色统计。CNN 后续要做的，就是不再手工写这些特征，而是从数据中自动学习更强的局部视觉特征。

## Slide 12：学习收获
这周主要学习了图像分类 pipeline、模型表达能力、超参数搜索、验证集和测试集的区别，以及输入表示对模型性能的影响。

## Slide 13：当前局限
现在的局限不再是没有跑 HOG/color histogram，因为这部分已经补上了。剩余局限包括：测试集还是 1,000 张，FullyConnectedNet 的 dropout 和 batch normalization 没有系统展开，还没有进入 CNN。

## Slide 14：下一步
下一步应该进入 Assignment 2：Dropout、BatchNorm、CNN 和 PyTorch。路线是从手工特征过渡到端到端自动学习图像特征。

## Slide 15：汇报结论
最后总结：Assignment 1 的价值不是只得到一个准确率，而是理解图像分类系统由数据、表示、模型、优化和调参共同决定。补充 features 后，Week5 已经覆盖 A1 的主要实验主线。
