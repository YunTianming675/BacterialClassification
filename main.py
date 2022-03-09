import os

import myUtil


def main():
    myUtil.files_rename(os.getcwd() + "\\dataset\\train\\image", "image_train_", ".jpg")
    myUtil.files_rename(os.getcwd() + "\\dataset\\test\\image", "image_test_", ".jpg")

    path = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02"
    savePath = r"D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\tailoring"
    img_list = os.listdir(path)
    for i in range(len(img_list)):
        img_path = path + "\\" + img_list[i]
        myUtil.image_tailoring(img_path, savePath, 300, 400)

    tailored_file_list = os.listdir(savePath)
    target_dir = r" D:\\Project\\VSCode\\Python\\ImageTest01\\dataset\\train_02\\not_black\\"
    for i in range(len(tailored_file_list)):
        if myUtil.is_all_black(savePath + "\\" + tailored_file_list[i], 30):
            continue
        else:
            copy_name = "copy " + savePath + "\\" + tailored_file_list[i] + target_dir
            if os.path.isfile(copy_name):
                continue
            else:
                os.system(copy_name)
    myUtil.files_rename(os.getcwd() + "\\dataset\\train_02\\not_black", "train_tailoring_", ".jpg")


if __name__ == "__main__":
    main()
