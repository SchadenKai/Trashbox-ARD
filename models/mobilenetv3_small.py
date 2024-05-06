# Load Depedencies

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torch.optim as optim
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchattacks
import matplotlib.pyplot as plt


class MobileNetV3_small(nn.Module):
    def __init__(self):
        super(MobileNetV3_small, self).__init__()

        self.mobilenet_v3_small = models.mobilenet_v3_small(weights=None)

        self.features_conv = self.mobilenet_v3_small.features

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=7, bias=True)
        )
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
print(model.features)