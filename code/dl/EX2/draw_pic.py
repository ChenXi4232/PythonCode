import matplotlib.pyplot as plt
import re

TrainLoss_pattern = r'Train Loss: (\d+\.\d+)'
ValLoss_pattern = r'Val Loss: (\d+\.\d+)'
Resuming_pattern = r'Resuming training from epoch (\d+)'
start_epoch = 0

# 读取txt文件内容
with open("./output/depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25.txt", "r") as file:
    lines = file.readlines()

# 找到中断重新开始的位置
resume_index = None
for idx, line in enumerate(lines):
    if "Resuming training from epoch" in line:
        resume_index = idx
        start_epoch = int(re.search(Resuming_pattern, line).group(1))
        break

epochs = []
train_losses = []
val_losses = []
counter = 0

for line in lines:
    if "Epoch" in line:
        epoch = int(line.split("[")[1].split("/")[0])
        epochs.append(epoch)
        train_loss = float(line.split("Train Loss: ")[1].split(",")[0])
        train_losses.append(train_loss)
        val_loss = float(line.split("Val Loss: ")[1].split(",")[0])
        val_losses.append(val_loss)
        counter += 1
        if counter == start_epoch:
            break


# 如果找到了中断重新开始的位置
if resume_index is not None:
    # 从中断重新开始的位置提取训练损失和验证损失
    for line in lines[resume_index:]:
        if "Epoch" in line:
            epoch = int(line.split("[")[1].split("/")[0])
            epochs.append(epoch)
            train_loss = float(line.split("Train Loss: ")[1].split(",")[0])
            train_losses.append(train_loss)
            val_loss = float(line.split("Val Loss: ")[1].split(",")[0])
            val_losses.append(val_loss)

    # 绘制曲线图
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(
        "./output/depth3-2_kernel3-1_dropout3-0.2_normData-bn_lrCLR0.1-0.6-up0.25.png")
    plt.show()
else:
    print("No resuming point found in the file.")
