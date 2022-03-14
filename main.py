import os

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
        myUtil.csv_to_yololabel(os.path.join(os.getcwd(), r"dataset\train\1st_train.csv"),
                                os.path.join(os.getcwd(), r"dataset\train\label"))

    test_ReadYOLO = False
    if test_ReadYOLO:
        dataset = readYOLO.ReadYOLO(os.path.join(os.getcwd(), r"dataset\train\image"),
                                    os.path.join(os.getcwd(), r"dataset\train\label"),
                                    trans=False)
        pic, target = dataset.__getitem__(0)
        print(pic.shape)
        print(target)

    background_processing = True
    if background_processing:
        imgs_path = os.path.join(os.getcwd(), r"dataset\TailoringNotAllBlack")
        img_path = os.path.join(imgs_path, "Bao_01_08.jpg")
        save_path = os.path.join(os.getcwd(), r"dataset\BackgroundProcessing")
        myUtil.background_processing(img_path, save_path, 50, (208, 216, 229))


if __name__ == "__main__":
    main()
