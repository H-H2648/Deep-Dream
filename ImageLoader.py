import matplotlib.image as mpimg
import torchvision.transforms as transforms
from PIL import Image
import torch

#loader = transforms.Compose([transforms.Resize((imageSize, imageSize)), transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#Constants for normalizing. We chose not to normalize it because it doesn't seem to make a difference
#IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
#IMAGENET_STD_NEUTRAL = [1, 1, 1]

#if length is larger than 500, then my computer cannot load the image with tensors
def setLength(x, curMax=500):
    return min(x, curMax)

#outputs the tensor version of the image
def loadImage(imgPath):
    img = Image.open(imgPath)
    loader = transforms.Compose([transforms.Resize((setMax(img.size[0]), setMax(img.size[1]))), transforms.ToTensor()])
    img = loader(img).unsqueeze(0)
    return (img.to(device))
