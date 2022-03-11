import cv2
import numpy
import os
import torch

from torch.utils.data import Dataset

class ReadYOLO(Dataset):

    def __init__(self, imgs_dir: str, label_path: str, model_type="classification",  trans=None, device=None):
        """
        Args:
            :param imgs_dir: 数据集所在文件夹
            :param label_path: 标签所在文件夹
            :param model_type: 需要读取的模型类型
                "classification": 分类
                "objectdetection": 目标检测
                "facedetection": 人脸检测
            :param trans: 是否进行图像增强
        """
        super(ReadYOLO, self).__init__()
        self.device = device
        self.model_type = model_type
        self.trans = trans
        self.imgs_dir = imgs_dir
        self.label_path = label_path
        self.labels = os.listdir(self.label_path)  # 获得标签文件
        self.imgs = os.listdir(self.imgs_dir)  # 获得训练图片文件
        self.imgs_name = list(map(lambda x: x.split(".")[0], self.imgs))  # 获得不带.jpg后缀的图片名称


    # 修改 len 方法，使其返回标签的个数
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, item):
        list_target = []
        img = self.imgs[list(map(lambda x: x == self.labels[item].split(".")[0], self.imgs_name)).index(True)]  # 通过标签寻找对应的图片是哪一张
        img_dir = os.path.join(self.imgs_dir, img)
        with open(os.path.join(self.label_path, self.labels[item]), "r") as fp:
            for line in fp.readlines():
                if len(line.strip("\n")) > 0:  # 去掉换行符，如果去掉后长度依然>0，则表示这一行不是空的
                    nums = line.strip().split(" ")  # strip() 不带参数表示将字符串最前和最后的空格去掉，然后以空格进行切片
                    li = [*map(lambda x: float(x), nums)]  # 将切片后的字符串转为float型并存于一个列表，此时得到了一行的数值
                    list_target.append(li)  # 将得到的数值保存
        if len(list_target) == 0:
            array_target = numpy.array([])
        else:
            array_target = numpy.concatenate(list_target, axis=0).reshape(len(list_target), -1)  # 将读到的内容拼接为array数组
        picture = cv2.imread(img_dir)  # array = [w, h, 3]
        if self.trans:
            pass  # TODO 需要增加一个trans函数
            return picture.unsqueeze(0).to(self.device), torch.from_numpy(array_target).to(self.device)
        else:
            return picture, array_target

