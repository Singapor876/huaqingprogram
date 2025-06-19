# 深度学习模型与数据处理笔记
## 1. 数据集处理
### 1.1 数据集划分
#### 1.1.1 划分方式
内容：使用train_test_split按比例划分训练集和验证集，确保数据分布均匀。  
```python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
```
#### 1.1.2 路径处理
内容：数据集路径需明确，训练集和验证集路径分别设置，便于后续操作。  
```python
train_dir = r'/image2/train'
val_dir = r'/image2/val'
```
### 1.2 数据集加载
#### 1.2.1 自定义数据集
内容：ImageTxtDataset类通过txt文件加载图片路径和标签，灵活处理数据。
```python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        self.data_dir = os.path.dirname(txt_path)
        self.imgs_path = []
        self.labels = []
        self.folder_name = folder_name
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_path, label = line.split()
            label = int(label.strip())
            self.labels.append(label)
            self.imgs_path.append(img_path)
```
#### 1.2.2 数据预处理
内容：包括调整图片大小、归一化等操作，确保输入模型的数据格式统一。
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
## 2. 神经网络模型
### 2.1 GoogLeNet模型
#### 2.1.1 Inception模块
内容：多分支结构，包含1x1、3x3、5x5卷积及池化，增强特征提取能力。
```python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_features):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
```
### 2.1.2 模型结构
内容：通过多个Inception模块堆叠，实现深层网络结构，提升分类性能。
```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
```
### 2.2 MobileNet_v2模型
#### 2.2.1 Inverted Residual模块
内容：先进行逐点卷积扩展通道，再进行深度可分离卷积，减少计算量。
```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
```
### 2.2.2 模型特点
内容：适用于移动设备，轻量化设计，保持较高准确率。
```python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.features = [nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )]
```
### 2.3 MogaNet模型
#### 2.3.1 简化卷积层
内容：采用普通卷积层构建，结构简洁，易于实现。
```python
class MogaNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MogaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
```
### 2.3.2 模型性能
内容：适合初学者理解和实践，可作为基础模型进行改进。
```python
self.layer1 = self._make_layer(64, 64, 2)
self.layer2 = self._make_layer(64, 128, 2, stride=2)
```
### 2.4 ResNet18模型
#### 2.4.1 残差结构
内容：解决深层网络训练难题，通过残差连接避免梯度消失。
```python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
```
#### 2.4.2 预训练模型
内容：使用预训练权重，快速迁移至新任务，提高训练效率。
```python
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.to(device)
```
## 3. 模型训练与测试
### 3.1 训练流程
#### 3.1.1 损失函数
内容：使用交叉熵损失函数，适合分类任务，衡量模型输出与真实标签的差异。
```python
criterion = nn.CrossEntropyLoss()
```
#### 3.1.2 优化器
内容：Adam优化器动态调整学习率，加速模型收敛。
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
### 3.2 测试流程
#### 3.2.1 准确率计算
内容：统计预测正确的样本数，计算模型在测试集上的准确率。
```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
```

#### 3.2.2 日志记录
内容：使用TensorBoard记录训练损失和测试准确率，便于可视化分析。
```python
writer = SummaryWriter("logs/resnet18")
writer.add_scalar("Train Loss", train_loss, epoch)
writer.add_scalar("Test Acc", test_acc, epoch)
```

## 4. 激活函数与数据可视化
### 4.1 ReLU激活函数
#### 4.1.1 特点
内容：非线性激活，加速训练，避免梯度消失。
```python
self.relu = torch.nn.ReLU()
```

#### 4.1.2 应用
内容：在卷积神经网络中广泛应用，提升模型表达能力。
```python
output = self.relu(input)
```

### 4.2 数据可视化
#### 4.2.1 TensorBoard
内容：可视化训练过程，直观展示模型性能变化。
```python
writer = SummaryWriter("sigmod_logs")
writer.add_images("input", imgs, global_step=step)
```

#### 4.2.2 输入输出对比
内容：对比输入图片和经过激活函数处理后的输出，理解网络行为。
```python
writer.add_images("output", output_sigmod, global_step=step)
```
## 5. 数据准备脚本
### 5.1 创建txt文件
#### 5.1.1 功能
内容：自动生成训练集和验证集的txt文件，记录图片路径和标签。
```python
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")
```

#### 5.1.2 使用方法
内容：指定数据集路径和输出txt文件名，方便后续加载数据。
```python
create_txt_file(r'/image2/train', 'train.txt')
create_txt_file(r'/image2/val', "val.txt")
```
