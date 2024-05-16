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
    def __init__(self, pre_trained='best_NORMAL--Mobilenetv3Small.v1_epoch40'):
        super(MobileNetV3_small, self).__init__()
        self.mobilenet_v3_small = models.mobilenet_v3_small(weights=None)
        self.mobilenet_v3_small.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=7, bias=True)
        )

        self.features_conv = self.mobilenet_v3_small.features

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=7, bias=True)
        )

        if pre_trained != '':
            checkpoint = torch.load(f'./best_trained_models/{pre_trained}.pth')

            if 'module' in list(checkpoint['net'].keys())[0]:
                new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['net'].items()}
                self.mobilenet_v3_small.load_state_dict(new_state_dict)
            else:
                self.mobilenet_v3_small.load_state_dict(checkpoint['net'])
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avg_pool(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
