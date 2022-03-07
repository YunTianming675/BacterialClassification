import os

def filesRename(path, startName, suffix):
    """文件批量重命名
    param path: 文件所在目录
    param startName: 批量重命名的前缀
    param suffix: 文件后缀名
    """
    count = 0
    num = 0
    fileList = os.listdir(path)

    for files in fileList:
        oldDir = os.path.join(path, files)

        if os.path.isdir(oldDir):
            continue

        string = str(num)
        if len(string) == 1:
            string = "0" + string

        newDir = os.path.join(path, startName + string + suffix)
        if newDir == oldDir:
            num += 1
            continue

        try:
            os.rename(oldDir, newDir)
        except FileExistsError:
            print(newDir + "已存在，跳过")
        count += 1
        num += 1
    
    print("完成，共修改了 " + str(count) + " 个文件")

