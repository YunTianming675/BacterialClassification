# encoding=GBK

import cv2
import os
import torch

import test

from model import VGG16
from utils import myUtil
from utils import readYOLO


def main():
    img_tailoring = False  # 图片裁剪
    if img_tailoring:
        imgs_path = os.path.join(os.getcwd(), r"dataset\test_01\ChangedImages")
        imgs = os.listdir(imgs_path)
        save_path = os.path.join(os.getcwd(), r"dataset\test_01\Tailoring")
        for i in range(len(imgs)):
            img_path = os.path.join(imgs_path, imgs[i])
            myUtil.image_tailoring(img_path, save_path, 224, 224)

    filter_all_black = False  # 从裁剪的图片中筛选出不是全黑的图片
    if filter_all_black:
        imgs_path = os.path.join(os.getcwd(), r"dataset\test_01\Tailoring")
        imgs = os.listdir(imgs_path)
        filter_save_path = os.path.join(os.getcwd(), r"dataset\test_01\NotAllBlack")
        for i in range(len(imgs)):
            if myUtil.is_all_black(os.path.join(imgs_path, imgs[i]), 60):
                continue
            else:
                command = "move " + imgs_path + "\\" + imgs[i] + " " + filter_save_path
                os.system(command)

    label_conversion = False  # 标签转换，csv -> yolo
    if label_conversion:
        myUtil.csv_to_yololabel(os.path.join(os.getcwd(), r"dataset\test\test.csv"),
                                os.path.join(os.getcwd(), r"dataset\test\label"))

    test_ReadYOLO = False  # 对Read_YOLO类的测试
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

    background_processing = False  # 二值法背景处理(对目标产生了干扰)
    if background_processing:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        save_path = os.path.join(os.getcwd(), r"dataset\BackgroundProcessing")
        imgs = os.listdir(imgs_path)
        for i in range(len(imgs)):
            img_path = os.path.join(imgs_path, imgs[i])
            myUtil.background_processing(img_path, save_path, (196, 208, 218))

    draw_graph = False  # 通过train时记录的loss和correct画出loss和correct曲线
    if draw_graph:
        loss_file = os.path.join(os.getcwd(), r"result\list_loss_1.txt")  # 保存loss记录的文件的路径
        c_file = os.path.join(os.getcwd(), r"result\correct_1.txt")  # 保存correct记录的文件的路径
        myUtil.draw(loss_file, c_file, os.path.join(os.getcwd(), r"result\result_1.jpg"))

    validate = False  # 加载train得的网络参数并查看准确率
    if validate:
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

    check_background = False  # 背景处理（这个边缘处效果不好）
    if check_background:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        imgs = os.listdir(imgs_path)
        img = myUtil.check_background(os.path.join(imgs_path, imgs[0]), 180)
        cv2.imwrite(os.path.join(os.getcwd(), r"dataset\BackgroundProcessing", imgs[0]), img)

    change_background = False  # 背景处理（这个方法的处理效果可以）
    if change_background:
        imgs_path = os.path.join(os.getcwd(), r"dataset\test_01\NotAllBlack")
        imgs = os.listdir(imgs_path)
        for i in range(len(imgs)):
            img = myUtil.change_background(os.path.join(imgs_path, imgs[i]))
            img.save(os.path.join(os.getcwd(), r"dataset\test_01\BackgroundProcess", imgs[i]))
        print("all finish")

    select_img = False  # 从处理好的图片中筛选一部分作为test，剩下的作为train
    if select_img:
        imgs_path = os.path.join(os.getcwd(), r"dataset\train\image")
        target_path = os.path.join(os.getcwd(), r"dataset\test\image")
        myUtil.random_select(imgs_path, target_path)

    test_t = True  # 对没有标签的图片做预测
    if test_t:
        pass


if __name__ == "__main__":
    main()
