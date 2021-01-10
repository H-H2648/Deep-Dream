import matplotlib.image as mpimg
import torchvision.transforms as transforms
from PIL import Image
import torch

#loader = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
#IMAGENET_STD_NEUTRAL = [1, 1, 1]

def setMax(x, curMax=500):
    return min(x, curMax)

def loadImage(imgPath):
    img = Image.open(imgPath)
    #loader = transforms.Compose([transforms.Resize((setMax(imgContent.size[0]), setMax(imgContent.size[1]))), transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255)), transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
    loader = transforms.Compose([transforms.Resize((setMax(img.size[0]), setMax(img.size[1]))), transforms.ToTensor()])
    img = loader(img).unsqueeze(0)
    return (img.to(device))
