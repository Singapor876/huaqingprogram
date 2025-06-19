# 《深度学习代码学习笔记》

## 一、代码 1：`train_alex.py`（基于 AlexNet 的图像分类训练代码）

### （一）代码功能
本代码实现了一个基于 AlexNet 架构的图像分类模型的训练过程，使用了自定义的数据集。

### （二）代码结构与关键点

1. **数据集加载**
   - 使用了自定义的 `ImageTxtDataset` 类来加载数据集，而不是标准的 CIFAR-10 数据集。
   - 数据集的路径和格式：
     - 图像路径和标签信息存储在 `train.txt` 文件中。
     - 图像存储在 `D:\dataset\image2\train` 文件夹中。
   - 数据预处理：
     - 使用 `transforms.Resize(224)` 将图像调整为 224x224 大小，以适应 AlexNet 的输入要求。
     - 使用 `transforms.RandomHorizontalFlip()` 进行随机水平翻转，增强数据多样性。
     - 使用 `transforms.ToTensor()` 将图像转换为张量。
     - 使用 `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` 对图像进行归一化处理，这些均值和标准差是 ImageNet 数据集的常用参数。

2. **模型定义**
   - 定义了一个简化的 AlexNet 模型：
     - 包含 5 个卷积层和 3 个全连接层。
     - 卷积层使用了 `MaxPool2d` 进行下采样。
     - 最后一层的输出是 10，表明这是一个 10 类分类任务。
   - 模型的输入是 3 通道的 RGB 图像。

3. **训练流程**
   - 使用了 `DataLoader` 来加载数据，批量大小为 64。
   - 使用了交叉熵损失函数（`CrossEntropyLoss`）和随机梯度下降优化器（`SGD`），学习率为 0.01，动量为 0.9。
   - 每 500 步记录一次训练损失，并使用 TensorBoard 进行可视化。
   - 每个 epoch 结束后，对测试集进行评估，计算整体损失和准确率。
   - 模型在每个 epoch 结束后保存为 `.pth` 文件。

4. **测试流程**
   - 在测试阶段，使用 `torch.no_grad()` 禁用梯度计算，以减少内存消耗并提高计算速度。
   - 计算测试集的总损失和准确率，并将结果记录到 TensorBoard 中。

### （三）学习重点
1. **自定义数据集的使用**
   - 学习如何使用自定义的数据集格式，特别是通过文本文件加载图像路径和标签的方式。
   - 掌握如何对数据进行预处理，以适应模型的输入要求。

2. **AlexNet 架构的理解**
   - 理解 AlexNet 的基本结构，包括卷积层、池化层和全连接层的作用。
   - 学习如何根据任务需求调整模型的输出层。

3. **训练与测试流程**
   - 掌握 PyTorch 中模型训练和测试的基本流程，包括数据加载、损失计算、优化器更新、模型评估和保存。
   - 学习如何使用 TensorBoard 进行训练过程的可视化。

4. **数据增强技术**
   - 学习如何通过数据增强（如随机水平翻转）来提高模型的泛化能力。

---

## 二、代码 2：`transformer.py`（基于 Transformer 的 Vision Transformer 模型代码）

### （一）代码功能
本代码实现了一个基于 Transformer 架构的 Vision Transformer（ViT）模型，用于处理序列化的图像数据。

### （二）代码结构与关键点

1. **模块定义**
   - **FeedForward 模块**：
     - 包含一个线性层、GELU 激活函数、Dropout 和另一个线性层。
     - 使用了 `LayerNorm` 进行归一化。
   - **Attention 模块**：
     - 实现了多头自注意力机制。
     - 使用了 `Softmax` 函数计算注意力权重。
     - 使用了 `rearrange` 和 `repeat` 函数来处理张量的形状。
   - **Transformer 模块**：
     - 包含多个 Transformer 层，每层包含一个注意力模块和一个前馈模块。
     - 使用了残差连接（`x = attn(x) + x` 和 `x = ff(x) + x`）。
   - **ViT 模型**：
     - 将输入的图像序列化为 patches，并通过 Transformer 模型进行处理。
     - 使用了位置嵌入（`pos_embedding`）和类别嵌入（`cls_token`）。
     - 最后通过一个全连接层输出分类结果。

2. **模型结构**
   - 输入是一个序列化的图像（`time_series`），形状为 `(batch_size, channels, seq_len)`。
   - 模型将输入序列化为 patches，每个 patch 的大小为 `patch_size`。
   - 使用 Transformer 模型对 patches 进行处理。
   - 最终输出分类结果，形状为 `(batch_size, num_classes)`。

3. **测试代码**
   - 创建了一个 ViT 模型实例，并输入了一个随机生成的张量（`time_series`）。
   - 输出的 logits 形状为 `(batch_size, num_classes)`，表明模型可以正常工作。

### （三）学习重点
1. **Transformer 架构的理解**
   - 学习 Transformer 的基本结构，包括多头自注意力机制、前馈网络和残差连接。
   - 理解 `LayerNorm` 和 `Dropout` 在 Transformer 中的作用。

2. **Vision Transformer（ViT）的实现**
   - 学习如何将 Transformer 应用于图像数据，通过将图像序列化为 patches 来处理。
   - 理解位置嵌入和类别嵌入的作用。
   - 掌握如何通过 Transformer 模型进行图像分类。

3. **einops 库的使用**
   - 学习如何使用 `einops` 库来简化张量操作，例如 `rearrange` 和 `repeat` 函数。

4. **模型的输入与输出**
   - 理解 ViT 模型的输入格式（序列化的图像）和输出格式（分类结果）。

---

## 三、总结
今天学习了两个深度学习模型的实现：
1. **`train_alex.py`**：基于 AlexNet 的图像分类模型，重点在于自定义数据集的使用、数据预处理、模型训练和测试流程。
2. **`transformer.py`**：基于 Transformer 的 Vision Transformer 模型，重点在于 Transformer 架构的理解、Vision Transformer 的实现以及 `einops` 库的使用。

通过这两个代码的学习，加深了对卷积神经网络（CNN）和 Transformer 架构的理解，同时也掌握了如何处理自定义数据集和使用数据增强技术。