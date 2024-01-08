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
from Metrics import Metrics

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_losses = []
# train_losses = train_losses.to('cpu')
train_psnr_values = []
# train_psnr_values = train_psnr_values.to('cpu')
train_ssim_values = []
# Instantiate the Frame Interpolation Model
model = Data_Aug_Model()
NUM_GPU = torch.cuda.device_count()
if NUM_GPU > 1:
	model = nn.DataParallel(model)
model = model.to(device)

ob = Metrics()
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
num_epochs = 30
for epoch in tqdm(range(num_epochs)):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    epoch_losses = []
    epoch_psnr_values = []
    epoch_ssim_values = []
    for frame1, frame2, target in tqdm(dataloader):
        # Concatenate input frames
        inputs = torch.cat((frame1, frame2), 1)
        inputs = inputs.to(device)
        # Zero the gradients, forward pass, calculate loss, backward pass, and optimizer step
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bicubic', align_corners=False).to(device)
        loss = criterion(outputs, target_resized).to(device)
        
        with torch.no_grad():
            psnr_value = ob.psnr(outputs, target_resized)
            ssim_value = ob.ssim(outputs, target_resized)
        
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            loss.detach().cpu()
            inputs.detach().cpu(), outputs.detach().cpu(), ssim_value.detach().cpu(),target_resized.detach().cpu()

        epoch_losses.append(loss.item())
        epoch_psnr_values.append(psnr_value)
        epoch_ssim_values.append(ssim_value)
        
        del inputs, outputs, psnr_value, ssim_value, loss, target_resized

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    avg_epoch_psnr = sum(epoch_psnr_values) / len(epoch_psnr_values)
    avg_epoch_ssim = sum(epoch_ssim_values) / len(epoch_ssim_values)
    
    train_losses.append(avg_epoch_loss)
    train_psnr_values.append(avg_epoch_psnr)
    train_ssim_values.append(avg_epoch_ssim)
 
    print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss}, Avg PSNR: {avg_epoch_psnr}, Avg SSIM: {avg_epoch_ssim}', flush=True)

torch.save(model.state_dict(), './Pre_trained_Model/frame_interpolation_model.pth')
metrics_data = {
    'train_losses': train_losses,
    'train_psnr_values': train_psnr_values,
    'train_ssim_values': train_ssim_values,
}

torch.save(metrics_data, './Plots/training_metrics.pth')
