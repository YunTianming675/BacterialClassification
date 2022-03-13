import argparse
import os
import time
import torch

from math import ceil
from model import VGG16
from torch.utils.data import DataLoader
from utils.data_augment import DataAugment
from utils.my_loss import MyLoss
from utils.readYOLO import ReadYOLO

parser = argparse.ArgumentParser(description="VGG16 Training")
parser.add_argument("--lr", default="0.001", help="learning rate of model")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
parser.add_argument("--epochs", default=20, type=int, help="epochs")
parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay for SGD")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据集
data_argument = DataAugment()
dataset = ReadYOLO(os.path.join(os.getcwd(), "dataset\\train_03"), os.path.join(os.getcwd(), "dataset\\yolo_label"), trans=data_argument, device=device)
picture_num = len(dataset)

# 模型实例化
net = VGG16()
net.train()
net = net.to(device=device)

# 迭代器和损失函数优化器实例化
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss = MyLoss()

# 创建图片数据迭代器
def colle(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.cat(imgs, dim=0)
    targets = torch.cat(targets, dim=0)
    return imgs, targets


# 样本数量不能被均分的话使用drop_last丢弃最后一批
data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)

def train():
    epochs = args.epochs
    batch_count = 0
    for epoch in range(epochs):
        for batch, (imgs, targets) in enumerate(data):
            start = time.time()
            pred = net(imgs)
            Loss = loss(pred, targets)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            # 训练完一个batch需要的时间
            batch_time = time.time() - start
            # 剩下的训练完需要的迭代次数
            total_num = epochs * ceil(picture_num / args.batch_size) - batch_count
            # 剩下的训练完需要的时间，返回的是秒
            rest_time = total_num * batch_time
            # 转化为 h/m/s
            hour = int(rest_time / 60 // 60)
            minute = int(rest_time // 60)
            second = int((rest_time / 60 - minute) * 60)
            batch_count += 1
            print("epoch:{0}/{1}".format(epoch + 1, epoch),
                  " ,loss: ", float(loss),
                  " ,每个batch所需时间：", batch_time,
                  " ,剩余批次：", total_num,
                  " ,eta: {0}小时{1}分钟{2}秒".format(hour, minute, second))
            if Loss <= 1e-3:
                torch.save(net.state_dict(), "./weights/An_Early_stop_params.pth".format(epoch + 1))
                return print("训练结束")
        # 每个 epoch 保存一次参数
        torch.save(net.state_dict(), "./weights/VGG16_epoch{}_params.pth".format(epoch + 1))
    print("训练结束")


if __name__ == "__main__":
    train()