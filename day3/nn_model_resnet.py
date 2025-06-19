import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

# 设置设备为CPU
device = torch.device("cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片大小调整为模型输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root="dataset_chen", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="dataset_chen", train=False, transform=transform, download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型
def load_resnet18():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 修改分类层以适配CIFAR-10的10个类别
    return model.to(device)

# 定义训练和测试函数
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 训练ResNet18模型
print("Training ResNet18...")
model = load_resnet18()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter("logs/resnet18")
for epoch in range(3):  # 训练3个epoch
    train_loss = train(model, train_dataloader, criterion, optimizer)
    test_acc = test(model, test_dataloader)
    print(f"Epoch [{epoch + 1}/3], Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")
    writer.add_scalar("Train Loss", train_loss, epoch)
    writer.add_scalar("Test Acc", test_acc, epoch)
writer.close()