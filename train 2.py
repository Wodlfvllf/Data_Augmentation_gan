import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from PIL import Image
import torch.nn.functional as F
from Frame_interpolation import Data_Aug_Model
from dataset import Custom_Dataset
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Instantiate the Frame Interpolation Model
model = Data_Aug_Model().to(device)
# Define the Mean Squared Error loss criterion
criterion = nn.MSELoss().to(device)
# Set up the Adam optimizer for model parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the Frame Interpolation Dataset
print("Loading dataset...")
dataset = Custom_Dataset('./Healthy/', transform=transforms.ToTensor())
# Create a DataLoader for batching and shuffling the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded.")

# Train the Frame Interpolation Model
print("Training model...")
num_epochs = 15
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for frame1, frame2, target in tqdm(dataloader):
        # Concatenate input frames
        inputs = torch.cat((frame1, frame2), 1)
        inputs = inputs.to(device)
        # Zero the gradients, forward pass, calculate loss, backward pass, and optimizer step
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bilinear', align_corners=False).to(device)
        loss = criterion(outputs, target_resized).to(device)
        loss.backward()
        optimizer.step()
    print(f'\nEpoch {epoch+1}/{num_epochs}, Loss: {loss.item()}', flush=True)

torch.save(model.state_dict(), './Pre_trained_Model/frame_interpolation_model.pth')