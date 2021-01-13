import torch
from VGG import VGGModel
from ImageLoader import loadImage
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
VModel = VGGModel().to(device)

imgDirectory = r'C:\Users\hhong\Documents\Deep-Dream\Images\giorno.jpg'
img = loadImage(imgDirectory).requires_grad_(True)

optimizer = optim.Adam([img], lr= 0.001)

#layer ranges from 0 to 4 (inclusive) corresponding tot he 5 convolutional channels of interest
def dreamify(layer):
    for step in range(1000):
        imgFeatures = VModel(img)
        #we take the gradient ascent
        #in this case we simply minize -1*f(x)
        gradient = -1*torch.nn.MSELoss(reduction='mean')(imgFeatures[layer], torch.zeros_like(imgFeatures[layer]))
        optimizer.zero_grad()
        gradient.backward()
        optimizer.step()
        if step % 200 == 0:
            print(gradient)
            save_image(img, "C:\\Users\\hhong\\Documents\\Deep-Dream\\Result11\\currentImg_{0}.jpg".format(int(step/200)))
    save_image(img, r"C:\Users\hhong\Documents\Deep-Dream\Result11\currentImg_FINAL.jpg")
        