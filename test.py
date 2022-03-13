import argparse
import cv2
import torch
import torchvision

from model import VGG16


parser = argparse.ArgumentParser(description="VGG16 Testing")
parser.add_argument("--weight_dir", default="./weights/An_Early_stop_params.pth", help="参数路径")
parser.add_argument("--test_dir", default="./dataset/test", help="测试图片路径")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
net = VGG16(mode="test")
net = net.to(device=device)

# 加载模型参数
net.load_state_dict(torch.load(args.weight_dir))

def run():
    net.eval()
    image = cv2.imread(args.test_dir, cv2.IMREAD_COLOR)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor()])
    image = trans(image).unsqueeze(0).to(device)
    result = torch.argmax(net(image).ravel())
    return result
