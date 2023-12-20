import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset

class LocalizationModel(nn.Module):
    def __init__(self, num_classes=10, train_CNN=False):
        super().__init__()
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')                    
        for param in resnet.parameters():
                param.requires_grad_(train_CNN)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.logits = nn.Linear(resnet.fc.in_features, num_classes)

        self.regressor = nn.Sequential(
             nn.Linear(resnet.fc.in_features, 128), nn.ReLU(),
             nn.Linear(128, 64), nn.ReLU(),
             nn.Linear(64, 32), nn.ReLU(),
             nn.Linear(32, 4),
             nn.Sigmoid()
        )

    def forward(self, x):
         res_features  = self.resnet(x)
         reshaped  = res_features.view(res_features.size(0), -1)
         logits = self.logits(reshaped)
         bbox = self.regressor(reshaped)

         return (logits, bbox)