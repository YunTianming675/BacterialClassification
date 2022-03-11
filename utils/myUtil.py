import cv2
import json
import os
import pandas

def files_rename(path: str, start_name: str, suffix: str):
    """文件批量重命名
    Args:
        :param path: 文件所在目录
        :param startName: 批量重命名的前缀
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
            if img[i][j][0] < threshold and img[i][j][1] < threshold and img[i][j][2] < 30:
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
            fp.write(img_label[i].split("[")[1].split("]")[0])
    print("generate yolo label finish, check")


def json_to_yololabel(json_path: str, save_path: str):
    """将json格式的数据转为yolo格式的数据
    Args:
        :param json_path: json 文件所在的路径
        :param save_path: 指定转换后yolo格式的保存路径
    """
    pass

