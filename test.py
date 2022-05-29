import os

import cv2
import torch
import torchvision

from utils import readYOLO


def run(net, imgs_path, label_path, device="cuda"):
    """测试，针对有标签的数据集，验证正确率
        Args:
            :param net: 网络模型
            :param imgs_path: 测试图片文件夹
            :param label_path: 测试图片对应标签的文件夹
            :param device: 运行设备选择
        Returns:
            float: 正确率
            judge: 判断结果列表
    """
    net.eval()
    imgs = os.listdir(imgs_path)
    dataset = readYOLO.ReadYOLO(imgs_path, label_path, trans=False, device=device)
    label_value = dataset.get_label_value()

    true_count = 0
    circle_count = 0
    judge = list()
    for i in range(len(label_value)):
        if dataset.imgs_name[i] == dataset.labels[i].split(".")[0]:
            image = cv2.imread(os.path.join(imgs_path, imgs[i]), cv2.IMREAD_COLOR)
            trans = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor()])
            image = trans(image).unsqueeze(0).to(device)
            result = torch.argmax(net(image).ravel())

            circle_count += 1
            string1 = str(result.data)
            if string1.split("(")[-1].split(",")[0] == str(label_value[i][0][0]):
                true_count += 1
                judge.append(True)
            else:
                judge.append(False)
        else:
            continue
    return true_count / circle_count, judge


def test(net, imgs_path, save, device="cuda"):
    """对没有标签的图片进行推理
        Args:
            :param net: 使用的模型
            :param imgs_path: 图片所在的目录
            :param save：推理结果保存路径
            :param device：使用的计算设备
    """
    net.eval()
    imgs = os.listdir(imgs_path)
    for i in range(len(imgs)):
        image = cv2.imread(os.path.join(imgs_path, imgs[i]), cv2.IMREAD_COLOR)
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor()])
        image = trans(image).unsqueeze(0).to(device)
        result = torch.argmax(net(image).ravel())
        print(result)

        string1 = str(result.data)
        label = string1.split("(")[-1].split(",")[0]
        image_name = imgs[i].split(".")[0] + ".txt"
        save_file = os.path.join(save, image_name)
        with open(save_file, "w") as file:
            file.write(label)
            file.close()
