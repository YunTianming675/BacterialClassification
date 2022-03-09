import os

import myUtil

myUtil.filesRename(os.getcwd() + "\\dataset\\train\\image", "image_train_", ".jpg")
myUtil.filesRename(os.getcwd() + "\\dataset\\test\\image", "image_test_", ".jpg")

path = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\BaoManBuDongGan_Yin_47.jpg"
savePath = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\tailoring"
myUtil.imageTailoring(path, savePath, 300, 400)
