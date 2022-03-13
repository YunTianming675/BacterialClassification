import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyLoss(nn.Module):
    # 相当于nn.CrossEntropyLoss
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, label, eps=1e-3):
        batch_size = pred.shape[0]  # 获取预测值的batch维度
        new_pred = pred.reshape(batch_size, -1)
        expand_target = torch.zeros(new_pred.shape, device=device)
        for i in range(batch_size):
            expand_target[i, int(label[i])] = 1  # BUG 标签类型需要修改
        softmax_pred = torch.softmax(new_pred, dim=1)
        return torch.sum(-torch.log(softmax_pred + eps) * expand_target) / batch_size

    # __call__ 方法可以不用重写，因为forward在父类中已重写
    def __call__(self, *args, **kwargs):
        return self.forward(*args)


if __name__ == "__main__":
    a = torch.tensor([[0.0791, -0.2797, 0.5169, -0.1229, 0.4389],
                      [-0.1366, 0.0622, 0.1356, 0.2589, 0.5595]], device=device)
    b = torch.tensor([[0], [1]], device=device)
    loss = MyLoss()
    my_loss = loss(a, b)
    print(my_loss)
