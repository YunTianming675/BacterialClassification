import os
import cv2

def files_rename(path: str, start_name: str, suffix: str):
    """文件批量重命名
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

