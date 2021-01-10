import torch
from VGG import VGGModel
from ImageLoader import loadImage
import torch.optim as optim
from torchvision.utils import save_image


device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
VModel = VGGModel().to(device)

imgDirectory = r'C:\Users\hhong\Documents\Deep-Dream\Images\giorno.jpg'
img = loadImage(imgDirectory).requires_grad_(True)
optimizer = optim.Adam([img], lr= 0.001)


#For now lower layers only
#Higher layers
for step in range(1000):
    imgFeatures = VModel(img)
    gradient = -1*torch.nn.MSELoss(reduction='mean')(imgFeatures[4], torch.zeros_like(imgFeatures[4]))
    optimizer.zero_grad()
    gradient.backward()
    optimizer.step()
    if step % 200 == 0:
        print(gradient)
        save_image(img, "C:\\Users\\hhong\\Documents\\Deep-Dream\\Result3\\currentImg_{0}.jpg".format(int(step/200)))
save_image(img, r"C:\Users\hhong\Documents\Deep-Dream\Result3\currentImg_FINAL.jpg")
    