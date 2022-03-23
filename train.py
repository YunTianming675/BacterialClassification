import argparse
import os
import torch

from math import ceil

import test
from model import VGG16
from torch.utils.data import DataLoader
from utils.data_augment import DataAugment
from utils.my_loss import MyLoss
from utils.readYOLO import ReadYOLO

parser = argparse.ArgumentParser(description="VGG16 Training")
parser.add_argument("--lr", default=0.001, help="learning rate of model")
parser.add_argument("--momentum", default=0.01, type=float, help="momentum")  # 初始权重
parser.add_argument("--batch_size", default=10, type=int, help="batch_size")  # batch_size:一个批次的大小
parser.add_argument("--epochs", default=20, type=int, help="epochs")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay for SGD")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# 读取数据集
data_argument = DataAugment()
dataset = ReadYOLO(os.path.join(os.getcwd(), r"dataset\train\image"),
                   os.path.join(os.getcwd(), r"dataset\train\label"),
                   trans=data_argument,
                   device=device)
picture_num = len(dataset)

# 模型实例化
net = VGG16()
net.train()
net = net.to(device=device)

# 迭代器和损失函数优化器实例化
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss = MyLoss()
loss.to(device=device)


# 创建图片数据迭代器
def collect(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.cat(imgs, dim=0)
    targets = torch.cat(targets, dim=0)
    return imgs, targets


# 样本数量不能被均分的话使用drop_last丢弃最后一批
data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collect)


def save_loss(list_loss: list, path: str):
    with open(path, "w") as file:
        file.write(str(list_loss))
        file.close()


def train():
    epochs = args.epochs
    batch_count = 0
    list_loss = list()
    loss_aver = list()
    correct = list()
    early_end = False
    average = picture_num / args.batch_size  # 除数因子，一次训练完成后的loss除以它，获得平均loss
    for epoch in range(epochs):
        for batch, (imgs, targets) in enumerate(data):
            pred = net(imgs)
            var_loss = loss(pred, targets)
            list_loss.append(float(var_loss))
            optimizer.zero_grad()
            var_loss.backward()
            optimizer.step()
            # 剩下的训练完需要的迭代次数
            total_num = epochs * ceil(picture_num / args.batch_size) - batch_count
            batch_count += 1
            print("epoch:{0}/{1}".format(epoch + 1, epochs),
                  " ,loss: ", float(var_loss),
                  " ,剩余批次：", total_num)
            if var_loss <= 1e-3:
                torch.save(net.state_dict(), "./weights/An_Early_stop_params.pth".format(epoch + 1))
                early_end = True
                # 测试，获得准确率
                net_test = VGG16(mode="test")
                net_test = net_test.to(device=device)
                net_test.load_state_dict(torch.load("./weights/An_Early_stop_params.pth"))
                c, j = test.run(net_test, os.path.join(os.getcwd(), r"dataset\test\image"),
                                os.path.join(os.getcwd(), r"dataset\test\label"))
                correct.append(c)
                sum_2 = 0
                for i in list_loss:
                    sum_2 += i
                loss_aver.append(sum_2 / batch)
                list_loss.clear()
                break
        # 每个 epoch 保存一次参数
        if early_end:
            early_end = False
        else:
            sum_l = 0
            for i in list_loss:
                sum_l += i
            loss_aver.append(sum_l / average)
            list_loss.clear()
            torch.save(net.state_dict(), "./weights/VGG16_epoch{}_params.pth".format(epoch + 1))
            # 测试，获得准确率
            net_test = VGG16(mode="test")
            net_test = net_test.to(device=device)
            net_test.load_state_dict(torch.load("./weights/VGG16_epoch{}_params.pth".format(epoch + 1)))
            c, j = test.run(net_test, os.path.join(os.getcwd(), r"dataset\test\image"),
                            os.path.join(os.getcwd(), r"dataset\test\label"))
            correct.append(c)
    print("训练结束")
    save_loss(loss_aver, os.path.join(os.getcwd(), r"result\list_loss_1.txt"))
    save_loss(correct, os.path.join(os.getcwd(), r"result\correct_1.txt"))


if __name__ == "__main__":
    train()
