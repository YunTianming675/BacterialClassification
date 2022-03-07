import os

import myUtil

myUtil.filesRename(os.getcwd() + "\\dataset\\train\\image", "image_train_", ".jpg")
myUtil.filesRename(os.getcwd() + "\\dataset\\test\\image", "image_test_", ".jpg")
