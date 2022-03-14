import torchvision


class DataAugment(object):
    def __init__(self):
        super(DataAugment, self).__init__()

    # 重写此方法，相当于重写()
    # 即可使用对象名()完成对__call__()函数的调用
    def __call__(self, *args, **kwargs):
        return self.detect_resize(*args)

    def detect_resize(self, img, label, size: tuple):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=size),
            torchvision.transforms.ToTensor()  # 将图片从numpy格式转换为tensor格式，会将值压缩至float[0.0 1.0]
        ])
        image = trans(img)
        return image, label
