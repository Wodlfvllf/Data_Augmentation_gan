
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from PIL import Image
import torch.nn.functional as F

class Custom_Dataset(Dataset):
    def __init__(self, video_folder, transform=None):
        # Initialization method where you define dataset parameters
        self.path = video_folder  # Corrected from 'path' to 'video_folder'
        self.transform = transform  # Optional image transformation (e.g., resizing, normalization)
        self.num_frames = len(os.listdir(video_folder))  # Total number of frames

    def __len__(self):
        # Define the length of the dataset, i.e., the total number of frames
        return self.num_frames - 2  # We subtract 2 since we need two consecutive frames to predict the third

    def __path(self, index):
        frame1_idx = index + 1
        frame2_idx = index + 2
        target_idx = index + 3

        path_1 = os.path.join(self.path, f"image_{frame1_idx:03d}.png")
        path_2 = os.path.join(self.path, f"image_{frame2_idx:03d}.png")
        path_3 = os.path.join(self.path, f"image_{target_idx:03d}.png")

        return path_1, path_2, path_3

    def __getitem__(self, idx):
        # Open and transform the frames using the provided transform (if any)
        path_1, path_2, path_3 = self.__path(idx)  # Corrected from self.path(idx) to self.__path(idx)

        frame1, frame2, target = map(Image.open, (path_1, path_2, path_3))
        if self.transform:
            frame1, frame2, target = map(self.transform, (frame1, frame2, target))

        return frame1, frame2, target

# Loading dataset
# dataset = Custom_Dataset('/Users/shashank/python/UIT_Challenge/Healthy/', transform=transforms.ToTensor())
