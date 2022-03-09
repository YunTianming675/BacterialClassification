import os

import myUtil


def main():
    myUtil.files_rename(os.getcwd() + "\\dataset\\train\\image", "image_train_", ".jpg")
    myUtil.files_rename(os.getcwd() + "\\dataset\\test\\image", "image_test_", ".jpg")

    path = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\BaoManBuDongGan_Yin_47.jpg"
    savePath = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\tailoring"
    myUtil.image_tailoring(path, savePath, 300, 400)


if __name__ == "__main__":
    main()
