import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from PIL import Image
import torch.nn.functional as F


class Data_Aug_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor module with three sets of convolutional layers
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                
            ) for i in range(3)
        ])
        # Upsampling layer
        self.resize = nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False)  # Adjust size to (200, 200)
        # Fusion convolution layer
        self.fusion_conv = nn.Conv2d(384, 128, 3, 1, 1)
        # Upsample convolution layer
        self.upsample_conv = nn.ConvTranspose2d(128, 3, 3, 2, 1, 1, 1)

    def forward(self, x):
        # Extract features from each input frame
        feature_maps = [extractor(x) for extractor in self.feature_extractor]
        # Resize the feature maps
        feature_maps_resized = [self.resize(fm) for fm in feature_maps]
        # Concatenate the resized feature maps
        x = torch.cat(feature_maps_resized, 1)
        # Apply fusion convolution and upsample convolution
        x = F.relu(self.fusion_conv(x))
        x = self.upsample_conv(x)
        return x
