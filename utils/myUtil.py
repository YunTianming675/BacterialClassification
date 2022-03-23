import cv2
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import random

from PIL import Image


def files_rename(path: str, start_name: str, suffix: str):
    """文件批量重命名
    Args:
        :param path: 文件所在目录
        :param start_name: 批量重命名的前缀
        :param suffix: 文件后缀名
    """
    count = 0
    num = 0
    file_list = os.listdir(path)

    for files in file_list:
        old_dir = os.path.join(path, files)

        if os.path.isdir(old_dir):
            continue

        string = str(num)
        if len(string) == 1:
            string = "0" + string

        new_dir = os.path.join(path, start_name + string + suffix)
        if new_dir == old_dir:
            num += 1
            continue

        try:
            os.rename(old_dir, new_dir)
        except FileExistsError:
            print(new_dir + "已存在，跳过")
        count += 1
        num += 1
    
    print("完成，共修改了 " + str(count) + " 个文件")


def image_tailoring(img_path: str, save_path: str, h_step: int, w_step: int):
    """将图片进行裁剪
    Args:
        :param img_path: 要裁剪的图片的完整路径
        :param save_path: 指定裁剪后文件的存放目录
        :param h_step: 裁剪后子图片的高度
        :param w_step: 裁剪后子图片的长度
    """
    img = cv2.imread(img_path)
    if img is None:
        print("Error:file not found")
        return
    
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    img_name = img_path.split("\\")[-1].split(".")[0]
    x_num = 1
    y_num = 1
    x_start = 0
    x_end = h_step
    y_start = 0
    y_end = w_step
    while x_num <= int(img.shape[0] / h_step):
        while y_num <= int(img.shape[1] / w_step):
            img_cropped = img[x_start:x_end, y_start:y_end]

            str_xnum = str(x_num)
            if len(str_xnum) == 1:
                str_xnum = "0" + str_xnum
            str_ynum = str(y_num)
            if len(str_ynum) == 1:
                str_ynum = "0" + str_ynum
            str_number = "_" + str_xnum + "_" + str_ynum

            img_savename = save_path + "\\" + img_name + str_number + ".jpg"
            if os.path.isfile(img_savename):
                os.remove(img_savename)
            cv2.imwrite(img_savename, img_cropped)

            y_start += w_step
            y_end += w_step
            y_num += 1
        x_start += h_step
        x_end += h_step
        y_start = 0
        y_end = w_step
        x_num += 1
        y_num = 1
    
    print("tailoring finished, check")


def is_all_black(img_path: str, threshold: int) -> bool:
    """判断图像是否是全黑的
    Args:
        :param img_path: 图像完整路径
        :param threshold: BGR的判断阈值
    Returns:
        result: 如果全黑，为True，否则为False
    Raises:
        FileNotFound: 如果不能读取图像，则抛出此异常
    """
    img = cv2.imread(img_path)
    if img is None:
        ex = Exception("FileNotFound")
        raise ex

    result = True
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0] < threshold and img[i][j][1] < threshold and img[i][j][2] < threshold:
                continue
            else:
                result = False
                break

        if not result:
            break
    return result


def csv_to_yololabel(csv_path: str, save_path: str):
    """将csv格式的标签转为yolo格式的标签
    Args:
        :param csv_path: csv文件的完整路径
        :param save_path: yolo标签的保存地址
    """
    csv_name = pandas.read_csv(csv_path, header=None)
    img_name = csv_name.iloc[:, 0]
    img_label = csv_name.iloc[:, 1]
    for i, name in enumerate(img_name):
        with open(os.path.join(save_path, name.split(".")[0] + ".txt"), "w") as fp:
            # 增加判断：make sense制作的标签数据格式读出来是str类型，而excel处理后则为numpy.int64类型
            val_type = str(type(img_label[i])).split("'")[1]
            if val_type == "str":
                fp.write(img_label[i].split("[")[1].split("]")[0])
            elif val_type == "numpy.int64":
                fp.write(str(img_label[i]))
    print("generate yolo label finish, check")


def json_to_yololabel(json_path: str, save_path: str):
    """将json格式的数据转为yolo格式的数据
    Args:
        :param json_path: json 文件所在的路径
        :param save_path: 指定转换后yolo格式的保存路径
    """
    # TODO 后续添加
    pass


def background_processing(img_path: str, save_path: str, pixel: tuple,
                          lower: tuple = (0, 0, 0), upper: tuple = (50, 240, 240)):
    """二值法将图片的无关背景去除
        Args:
            :param img_path: 图像路径
            :param save_path: 修改后图像的保存路径文件夹
            :param pixel: 要替换的像素值，（B、G、R）格式
            :param lower: 二值处理的下边界（R、G、B）格式
            :param upper: 二值处理的上边界（R、G、B）格式
        Raises:
            FileNotFound: 如果读取文件失败，则抛出此异常
            WriteFileError: 如果图片保存失败，则抛出此异常
    """
    img_name = img_path.split("\\")[-1]
    img = cv2.imread(img_path)
    if img is None:
        ex = Exception("FileNotFound")
        raise ex
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = numpy.array(lower)
    upper = numpy.array(upper)
    mask = cv2.inRange(hsv, lower, upper)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i, j] == 0:
                img[i][j][0] = pixel[0]
                img[i][j][1] = pixel[1]
                img[i][j][2] = pixel[2]
    if not cv2.imwrite(os.path.join(save_path, img_name), img):
        ex = Exception("WriteFileError")
        raise ex
    print(img_name, "process finish")


def check_background(img_path: str, threshold: int, pixel: tuple = (226, 243, 247)) -> numpy.ndarray:
    """将图片中小于给定阈值的像素点以给定像素值进行替换
        Args:
            :param img_path: 图片的路径
            :param threshold: 阈值
            :param pixel: 替换像素值（BGR）格式
        Returns:
            替换后的图片，cv2格式（BGR）
        Raise:
            FileNotFound: 读取图片失败抛出此异常
    """
    img = cv2.imread(img_path)
    if img is None:
        ex = Exception("FileNotFound")
        raise ex
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0] < threshold and img[i][j][1] < threshold and img[i][j][2] < threshold:
                img[i][j][0] = pixel[0]
                img[i][j][1] = pixel[1]
                img[i][j][2] = pixel[2]
    img_name = img_path.split("\\")[-1]
    print(img_name, "process finish")
    return img


def change_background(img_path: str, black_threshold: int = 60, num_colors: int = 50,
                      change_threshold: tuple = (180, 180, 180),
                      select_threshold: tuple = (225, 225, 225),
                      default: tuple = (247, 243, 226)) -> Image.Image:
    """提取背景色中RGB最高的像素值并以此像素值对阈值下的像素进行覆盖
        Args:
            :param img_path: 图片路径
            :param black_threshold: 对黑色像素的判断阈值
            :param num_colors: 提取背景主要颜色时，选择要提取多少个点
            :param change_threshold: 被修改的像素的判断阈值，（RGB）
            :param select_threshold: 从提取到的背景主要颜色中选择像素值最高的点，此参数控制选择的最低像素值（RGB）
            :param default: 当没能找到符合要求的点时，默认以此像素值进行替换（RGB）
        Returns:
            :Image.Image: 修改后的图片
        Raise:
            FileNotFound: 读取图片失败抛出此异常
        @author Huiwen Zhou
    """
    try:
        image = Image.open(img_path)
    except FileNotFoundError:
        ex = Exception("FileNotFound")
        raise ex
    small_image = image.resize((100, 100))
    result = small_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    result = result.convert('RGB')
    main_colors = result.getcolors(maxcolors=256)
    max_color = 0
    for count, col in main_colors:
        if (col[0] < black_threshold) and (col[1] < black_threshold) and (col[2] < black_threshold):  # 剔除黑色
            continue

        elif (col[0] > select_threshold[0]) and (col[1] > select_threshold[1]) and (col[2] > select_threshold[2]):
            if count > max_color:
                default = col
                max_color = count  # 获取图片中RGB值最高的索引
    img_width = image.size[0]  # 获取图片宽度
    img_high = image.size[1]  # 获取图片长度
    col_extract2 = list(default)  # 背景色rgb值
    for i in range(img_width):
        j = 0
        while j < img_high:
            r, g, b = image.getpixel((i, j))
            if (r < change_threshold[0]) and (g < change_threshold[1]) and (b < change_threshold[2]):
                r = col_extract2[0]
                g = col_extract2[1]
                b = col_extract2[2]
                image.putpixel((i, j), (r, g, b))
                j = j + 1
            else:
                j = img_high

    # 图片翻转，修复部分黑色背景不被修改的问题
    image1 = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_width = image.size[0]  # 获取图片宽度
    img_high = image.size[1]  # 获取图片长度
    col_extract2 = list(default)  # 背景色rgb值
    for i in range(img_width):
        j = 0
        while j < img_high:
            r, g, b = image1.getpixel((i, j))
            if (r < change_threshold[0]) and (g < change_threshold[1]) and (b < change_threshold[2]):
                r = col_extract2[0]
                g = col_extract2[1]
                b = col_extract2[2]
                image1.putpixel((i, j), (r, g, b))
                j = j + 1
            else:
                j = img_high
    img_name = img_path.split("\\")[-1]
    print(img_name, "change finish")
    return image1.transpose(Image.FLIP_TOP_BOTTOM)


def random_select(imgs_path: str, target_path: str):
    """

    """
    imgs = os.listdir(imgs_path)
    imgs_bao = list()
    imgs_nian = list()
    imgs_shi = list()
    for i in range(len(imgs)):
        name = imgs[i].split("_")[0]
        if name == "Bao":
            imgs_bao.append(imgs[i])
        elif name == "Nian":
            imgs_nian.append(imgs[i])
        elif name == "Shi":
            imgs_shi.append(imgs[i])

    select_bao = random.sample(imgs_bao, 96)
    select_nian = random.sample(imgs_nian, 128)
    select_shi = random.sample(imgs_shi, 96)

    for i in range(len(select_bao)):
        os.system("move " + os.path.join(imgs_path, select_bao[i]) + " " + os.path.join(target_path, select_bao[i]))
    for i in range(len(select_nian)):
        os.system("move " + os.path.join(imgs_path, select_nian[i]) + " " + os.path.join(target_path, select_nian[i]))
    for i in range(len(select_shi)):
        os.system("move " + os.path.join(imgs_path, select_shi[i]) + " " + os.path.join(target_path, select_shi[i]))
    print("select finish")


def draw(loss_file: str, c_file:str, save: str):
    with open(loss_file, "r") as file:
        text = file.read()
        file.close()
    string1 = text.split(",")
    list_val = list()
    list_val.append(float(string1[0].split("[")[1]))
    for i in range(len(string1)):
        if i == 0 or i == len(string1) - 1:
            continue
        list_val.append(float(string1[i]))
    list_val.append(float(string1[-1].split("]")[0]))
    list_val.append(float(string1[-1].split("]")[0]))

    with open(c_file, "r") as file:
        text = file.read()
        file.close()
    string2 = text.split(",")
    list_c = list()
    list_c.append(float(string2[0].split("[")[1]))
    for i in range(len(string2)):
        if i == 0 or i == len(string2) - 1:
            continue
        list_c.append(float(string2[i]))
    list_c.append(float(string2[-1].split("]")[0]))
    list_c.append(float(string2[-1].split("]")[0]))

    x = list(range(1, len(list_c) + 1))
    plt.plot(list_val, label="loss")
    plt.plot(list_c, color="red", label="correct")
    plt.legend(loc="best", fontsize=6)
    plt.xticks(x)
    plt.savefig(save)
    plt.show()

