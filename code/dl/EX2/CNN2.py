import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
# import torchvision.transforms.functional as F


resume_training = False
save_dir = './save_state/'

# 创建保存状态的文件夹
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 设置随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stdout_backup = sys.stdout
if resume_training is False:
    sys.stdout = open('output.txt', 'w')
else:
    sys.stdout = open('output.txt', 'a')


def save_checkpoint(epoch, model, optimizer, scheduler, ES_counter=0):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ES_counter': ES_counter
    }
    torch.save(state, os.path.join(save_dir, 'checkpoint.pth'))


# def save_checkpoint(epoch, model, optimizer, ES_counter=0):
#     state = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'ES_counter': ES_counter
#     }
#     torch.save(state, os.path.join(save_dir, 'checkpoint.pth'))


def load_checkpoint(model, optimizer, scheduler):
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
    epoch = checkpoint['epoch']
    ES_counter = checkpoint['ES_counter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return epoch + 1, ES_counter  # 恢复训练时从下一个epoch开始


# def load_checkpoint(model, optimizer, scheduler):
#     checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pth'))
#     epoch = checkpoint['epoch']
#     ES_counter = checkpoint['ES_counter']
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     return epoch + 1, ES_counter  # 恢复训练时从下一个epoch开始


def print_and_write(*args, **kwargs):
    # Write to file and print to terminal
    print(*args, **kwargs, file=stdout_backup)
    print(*args, **kwargs)
    sys.stdout.flush()  # 确保立即写入文件


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0

        mask = torch.from_numpy(mask)
        img *= mask.unsqueeze(0)

        return img


class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth', ES_counter=0):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = ES_counter
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print_and_write(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
            print_and_write(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')


# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True)

data = trainset.data / 255.0

mean_CIFAR10 = data.mean(axis=(0, 1, 2))
std_CIFAR10 = data.std(axis=(0, 1, 2))

# 定义数据增强的transform
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
    Cutout(n_holes=8, length=32),
    transforms.ToTensor(),
#     transforms.Normalize(mean=mean_CIFAR10, std=std_CIFAR10),
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
])

transform = transforms.Compose([
    transforms.ToTensor(),
#     transforms.Normalize(mean=mean_CIFAR10, std=std_CIFAR10)
])

# 划分训练集和验证集
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

train_dataset.dataset.transform = transform_train
val_dataset.dataset.transform = transform
testset.transform = transform

# 创建数据加载器
trainloader = DataLoader(train_dataset, batch_size=768, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=768, shuffle=False)
testloader = DataLoader(testset, batch_size=768, shuffle=False)

# 定义类别标签
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# 定义基本的卷积块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        # self.in1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # self.in2 = nn.BatchNorm2d(out_planes)
        # self.celu = nn.CELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.in1(out)
        # out = self.celu(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.in2(out)
        # out = self.celu(out)
        out = self.relu(out)
        out = torch.add(out, self.shortcut(x))
        return out


# 定义残差网络
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.CELU(inplace=True)
        )
        self.lay1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.BatchNorm2d(128),
            # nn.CELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res1 = BasicBlock(128, 128)
        self.lay2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.BatchNorm2d(256),
            # nn.CELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lay3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            # nn.BatchNorm2d(512),
            # nn.CELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res2 = BasicBlock(512, 512)
        self.lay4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            # nn.CELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lay5 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            # nn.CELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.res3 = BasicBlock(2048, 2048)
        self.dropout1 = nn.Dropout(0.6)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 512)
        self.dropout3 = nn.Dropout(0.6)
        self.linear = nn.Linear(512, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.prep(x)
        out = self.lay1(out)
        out = self.res1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.res2(out)
        # out = self.lay4(out)
        # out = self.lay5(out)
        # out = self.res3(out)
        out = self.dropout1(out)
        out = self.pool(out)
        out = self.flatten(out)
        # out = self.fc1(out)
        # out = self.dropout2(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.dropout3(out)
        # out = self.relu(out)
        out = self.linear(out)
        return out


# torch.autograd.set_detect_anomaly(True)

# 实例化模型
model = ResNet().to(device)
summary(model, (3, 224, 224))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)

# scheduler = optim.lr_scheduler.StepLR(
#     optimizer, step_size=1, gamma=1.0)  # StepLR 调度器

# # 定义学习率预热阶段的 epoch 数和最大学习率
# warmup_epochs = 20
# max_lr = 0.2

# # 训练模型
# for epoch in range(warmup_epochs):
#     model.train()
#     for batch_idx, (inputs, labels) in enumerate(trainloader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     # 学习率逐步增加
#     current_lr = max_lr * (epoch + 1) / warmup_epochs
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = current_lr

#     print_and_write(
#         f"Epoch [{epoch+1}/{warmup_epochs}], Current Learning Rate: {current_lr}")

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# 训练模型
num_epochs = 200
train_losses = []
val_losses = []
val_accuracies = []
ES_counter = 0

up_ratio = 0.25
patience_ratio = 0.35
step_ratio = 0.2

# scheduler = optim.lr_scheduler.StepLR(
#     optimizer, step_size=step_ratio*num_epochs, gamma=0.5)

scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.1, max_lr=0.6, step_size_up=up_ratio*num_epochs,
    step_size_down=(1-up_ratio)*num_epochs)
# 加载检查点来恢复训练
if resume_training and os.path.exists(os.path.join(save_dir, 'checkpoint.pth')):
    start_epoch, ES_counter = load_checkpoint(model, optimizer, scheduler)
#     start_epoch, ES_counter = load_checkpoint(model, optimizer)
    print_and_write(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

early_stopping = EarlyStopping(patience=patience_ratio*num_epochs,
                               delta=0, path='ResNet9_best_model.pth', ES_counter=ES_counter)

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(trainloader))

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_losses.append(val_loss / len(valloader))
    val_accuracies.append(correct / total)

    # 检查是否早停
    early_stopping(val_losses[-1], model)
    ES_counter = early_stopping.counter
    # if early_stopping.early_stop:
    #     if train_losses[-1] <= 0.5:
    #         print_and_write("Early stopping")
    #         break
    #     else:
    #         print_and_write(
    #             "Early stopping is not triggered for not converging. Resetting the counter.")
    #         early_stopping.early_stop = False
    #         early_stopping.counter = 0

    print_and_write(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]*100:.2f}%")

    scheduler.step()

    # 每隔总轮数的10%进行一次当前模型参数、优化器参数、学习率调度器参数、epoch轮数的保存
    if (epoch + 1) % (num_epochs // 10) == 0:
        save_checkpoint(epoch, model, optimizer, scheduler, ES_counter)
#         save_checkpoint(epoch, model, optimizer, ES_counter)

# 可视化训练过程中的损失
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig('loss_plot.png')
# plt.show()

# 在训练结束后加载最佳模型
model.load_state_dict(torch.load('ResNet9_best_model.pth'))

# 在测试集上评估模型
test_correct = 0
test_total = 0
test_loss = 0.0

model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

if not os.path.exists('./ul_output/'):
    os.makedirs('./ul_output/')

print_and_write(
    f"Test Loss: {test_loss / len(testloader):.4f}, Test Acc: {test_correct / test_total * 100:.2f}%")

# Restoring stdout
sys.stdout.close()
sys.stdout = stdout_backup

file_name_prefix = 'depth3-2_kernel3-1_dropout1-0.6_NotnormData-bn_lrCLR0.1-0.6-up0.25_Aug'

if os.path.exists('./ul_output/'+file_name_prefix+'.txt'):
    os.remove('./ul_output/'+file_name_prefix+'.txt')
# Save the output text file to the output directory
os.rename('output.txt', './ul_output/'+file_name_prefix+'.txt')

if os.path.exists('./ul_output/'+file_name_prefix+'.png'):
    os.remove('./ul_output/'+file_name_prefix+'.png')
# Save the loss plot to the output directory
os.rename('loss_plot.png', './ul_output/'+file_name_prefix+'.png')
