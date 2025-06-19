import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageTxtDataset


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(48, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.Conv2d(192, 128, kernel_size=3, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 4 * 4, 2048),
            torch.nn.Linear(2048, 2048),
            torch.nn.Linear(2048, 10)
        )

    def forward(self, x):
        return self.model(x)


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transform)
# test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transform)

train_data = ImageTxtDataset(r"D:\dataset\train.txt",
                             r"D:\dataset\image2\train",
                             transforms.Compose([transforms.Resize(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )

test_data = ImageTxtDataset(r"D:\dataset\train.txt",
                             r"D:\dataset\image2\train",
                             transforms.Compose([transforms.Resize(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = AlexNet()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

total_train_step = 0
total_test_step = 0
epoch = 10
writer = SummaryWriter("../logs_train")

start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    model.train()
    for data in train_loader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / len(test_data)}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

    torch.save(model, f"model_save\\alexnet_{i}.pth")
    print("模型已保存")

writer.close()