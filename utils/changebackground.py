from PIL import Image


def change_background(img_path: str):
    """提取背景色中RGB最高的像素值并以此像素值对阈值下的像素进行覆盖

    """
    try:
        image = Image.open(img_path)
    except FileNotFoundError:
        print("找不到指定的图片")
        return
    # 要提取的主要颜色数量
    num_colors = 50
    small_image = image.resize((100, 100))
    result = small_image.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    result = result.convert('RGB')
    main_colors = result.getcolors(maxcolors=256)
    max_color = 0
    for count, col in main_colors:
        if (col[0] < 60) & (col[1] < 60) & (col[2] < 60):  # 剔除黑色
            continue

        elif (col[0] > 225) & (col[1] > 225) & (col[2] > 225):
            if count > max_color:
                col_extract1 = col
                max_color = count  # 获取图片中RGB值最高的索引
    image1 = image
    img_width = image1.size[0]  # 获取图片宽度
    img_high = image1.size[1]  # 获取图片长度
    col_extract2 = list(col_extract1)  # 背景色rgb值
    for i in range(img_width):
        j = 0
        while j < img_high:
            r, g, b = image.getpixel((i, j))
            if (r < 220) & (g < 220) & (b < 220):
                r = col_extract2[0]
                g = col_extract2[1]
                b = col_extract2[2]
                image1.putpixel((i, j), (r, g, b))
                j = j + 1
            else:
                j = img_high

    for i in range(img_width):
        for j in range(img_high):
            r, g, b = image1.getpixel((i, j))
            if (b > 250) & (r > 250) & (g > 250):
                r = col_extract2[0]
                g = col_extract2[1]
                b = col_extract2[2]
                image1.putpixel((i, j), (r, g, b))
    return image1


# img_path = 'C:/Users/hfbd/Desktop/test1/hunhejun/smy/smy1.jpg'
# image = Image.open(img_path)
# image2 = change_background(image)  # 更换背景
# img1 = transforms.RandomVerticalFlip(p=1)(image2)  # 垂直翻转
# # img1=transforms.RandomHorizontalFlip(p=1)(image2)#水平翻转
# image3 = change_background(img1)
# image3.save('C:/Users/hfbd/Desktop/test1/hunhejun/smy/18.jpg')
