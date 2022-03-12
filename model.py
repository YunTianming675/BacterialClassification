import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 前 3 层中是没有 1x1 卷积的，故这里默认False
def VGG_block(input_channel, output_channel, use_1x1=False):
    """构建VGG块
    """
    block = [nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), padding=1, device=device),
             nn.BatchNorm2d(output_channel, device=device),
             nn.ReLU(),
             nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), padding=1, device=device),
             nn.BatchNorm2d(output_channel, device=device),
             nn.ReLU()]
    if use_1x1:
        block.append(nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(1, 1), padding=1, device=device))
        block.append(nn.BatchNorm2d(output_channel, device=device))
        block.append(nn.ReLU())
        block.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    else:
        block.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    return block


class VGG16(nn.Module):
    def __init__(self, inputs=64, block_count=4, mode="train"):
        super().__init__()
        self.mode = mode
        self.block_count = block_count
        self.blocki = []
        self.block0 = VGG_block(input_channel=3, output_channel=inputs, use_1x1=False)  # 3 通道转 64 通道
        self.blocki.extend(self.block0)  # entend方法将追加的列表中的值一个一个放入目标列表，（extend和append的不同
        # setattr方法给对象添加变量，并指定变量的值
        # 字符串可以传值
        for count in range(1, block_count):
            setattr(self, "block{}".format(count), VGG_block(input_channel=inputs, output_channel=inputs * 2, use_1x1=(True if count != 1 else False)))
            inputs = 2 * inputs
            self.output = inputs
        for count in range(1, self.block_count):
            self.blocki.extend(getattr(self, "block{}".format(count)))
        self.block4 = VGG_block(input_channel=self.output, output_channel=self.output, use_1x1=True)
        self.blocki.extend(self.block4)
        self.block5 = [
            nn.Conv2d(in_channels=self.output, out_channels=self.output // 2, kernel_size=(3, 3), device=device),
            nn.BatchNorm2d(self.output // 2, device=device),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output // 2, out_channels=5, kernel_size=(6, 6))]
        self.blocki.extend(self.block5)
        self.result = nn.Sequential(*self.blocki)

    def forward(self, x):
        if self.mode == "test":
            return self.result(x).softmax(dim=1)
        else:
            return self.result(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VGG16(mode="test")
    net = net.to(device=device)
    x = torch.randn(2, 3, 224, 224, device=device)

    from torchsummary import summary
    summary(net, (3, 224, 224))

    # print(net(x))
