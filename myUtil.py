import os
import cv2

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


def imageTailoring(imgPath, savePath, xStep, yStep):
    img = cv2.imread(imgPath)
    if img is None:
        print("Error:file not found")
        return
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        pass

    imgName = imgPath.split("\\")[-1].split(".")[0]
    xNum = 1
    yNum = 1
    xStart = 0
    xEnd = xStep
    yStart = 0
    yEnd = yStep
    while xNum <= int(img.shape[0] / xStep):
        while yNum <= int(img.shape[1] / yStep):
            imgCropped = img[xStart:xEnd, yStart:yEnd]

            stringXNum = str(xNum)
            if len(stringXNum) == 1:
                stringXNum = "0" + stringXNum
            stringYNum = str(yNum)
            if len(stringYNum) == 1:
                stringYNum = "0" + stringYNum
            stringNumber = "_" + stringXNum + "_" + stringYNum

            imgSaveName = savePath + "\\" + imgName + stringNumber + ".jpg"
            if os.path.isfile(imgSaveName):
                os.remove(imgSaveName)
            cv2.imwrite(imgSaveName, imgCropped)

            yStart += yStep
            yEnd += yStep
            yNum += 1
        xStart += xStep
        xEnd += xStep
        yStart = 0
        yEnd = yStep
        xNum += 1
        yNum = 1
    
    print("tailoring finished, check")

