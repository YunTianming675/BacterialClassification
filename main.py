# encoding=GBK

import cv2
import matplotlib.pyplot as plt
import os
import torch

import test

from model import VGG16
from utils import myUtil
from utils import readYOLO


def main():
    img_tailoring = False
    if img_tailoring:
        imgs_path = os.path.join(os.getcwd(), r"dataset\changedImages")
        imgs = os.listdir(imgs_path)
        tail_save_path = os.path.join(os.getcwd(), r"dataset\changedImagesTailoring")
        for i in range(len(imgs)):
            img_path = os.path.join(imgs_path, imgs[i])
            myUtil.image_tailoring(img_path, tail_save_path, 224, 224)

    filter_all_black = False
    if filter_all_black:
        imgs_path = os.path.join(os.getcwd(), r"dataset\changedImagesTailoring")
        imgs = os.listdir(imgs_path)
        filter_save_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        for i in range(len(imgs)):
            if myUtil.is_all_black(os.path.join(imgs_path, imgs[i]), 50):
                continue
            else:
                command = "copy " + imgs_path + "\\" + imgs[i] + " " + filter_save_path
                os.system(command)

    label_conversion = False
    if label_conversion:
        myUtil.csv_to_yololabel(os.path.join(os.getcwd(), r"dataset\train\1000.csv"),
                                os.path.join(os.getcwd(), r"dataset\train\label"))

    test_ReadYOLO = False
    if test_ReadYOLO:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = readYOLO.ReadYOLO(os.path.join(os.getcwd(), r"dataset\test\image"),
                                    os.path.join(os.getcwd(), r"dataset\test\label"),
                                    trans=False,
                                    device=device)
        pic, target = dataset.__getitem__(0)
        print(pic.shape)
        print(target)
        label_value = dataset.get_label_value()
        print(type(label_value[0][0]))
        print(label_value[1][0][0])

    background_processing = False  # 对目标产生了干扰
    if background_processing:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        save_path = os.path.join(os.getcwd(), r"dataset\BackgroundProcessing")
        imgs = os.listdir(imgs_path)
        for i in range(len(imgs)):
            img_path = os.path.join(imgs_path, imgs[i])
            myUtil.background_processing(img_path, save_path, (196, 208, 218))

    draw_graph = True
    if draw_graph:
        file_path = os.path.join(os.getcwd(), r"result\list_loss_2.txt")  # 保存loss记录的文件的路径
        # 读取保存的loss
        with open(file_path, "r") as file:
            text = file.read()
            file.close()
        # 将loss转换为list，list元素类型是float
        string1 = text.split(",")
        list_loss = list()
        list_loss.append(float(string1[0].split("[")[1]))
        for i in range(len(string1)):
            if i == 0 or i == len(string1) - 1:
                continue
            list_loss.append(float(string1[i]))
        list_loss.append(float(string1[-1].split("]")[0]))
        plt.plot(list_loss)
        plt.savefig(os.path.join(os.getcwd(), r"result\loss_2.jpg"))
        plt.show()

    test_test = True
    if test_test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载模型
        net = VGG16(mode="test")
        net = net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load("./weights/An_Early_stop_params.pth"))
        correct, judge = test.run(net, os.path.join(os.getcwd(), r"dataset\test\image"),
                                  os.path.join(os.getcwd(), r"dataset\test\label"))
        print(correct)
        print(judge)

    check_background = False
    if check_background:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        imgs = os.listdir(imgs_path)
        img = myUtil.check_background(os.path.join(imgs_path, imgs[0]), 180)
        cv2.imwrite(os.path.join(os.getcwd(), r"dataset\BackgroundProcessing", imgs[0]), img)

    change_background = False
    if change_background:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        imgs = os.listdir(imgs_path)
        for i in range(len(imgs)):
            img = myUtil.change_background(os.path.join(imgs_path, imgs[i]))
            img.save(os.path.join(os.getcwd(), r"dataset\BackgroundProcessing", imgs[i]))
        print("all finish")

    select_img = False
    if select_img:
        imgs_path = os.path.join(os.getcwd(), r"dataset\train\image")
        target_path = os.path.join(os.getcwd(), r"dataset\test\image")
        myUtil.random_select(imgs_path, target_path)


if __name__ == "__main__":
    main()
