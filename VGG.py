from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models

#These corresponds to the layer conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
focusConv = ['0', '5', '10', '19', '28']
#use gpu
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.focusConv = focusConv
        #we go up to 29 because we don't actually need the linear section (no need for actual prediction)
        self.model= (models.vgg19(pretrained=True).features[:29]).to(device)

    def forward(self, x):
        features = []
        #go up to the final convolutional layer
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            #store the reuslt of output of the desired convolutional layer
            if str(layer_num) in self.focusConv:
                features.append(x)
        return features