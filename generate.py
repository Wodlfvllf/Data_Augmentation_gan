# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torch.nn.functional import interpolate as F_interpolate
# from PIL import Image
# import torch.nn.functional as F
# from Frame_interpolation import Data_Aug_Model
# from dataset import Custom_Dataset
# from tqdm import tqdm
# import numpy as np

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')


# paths = []
# for i in os.listdir('./Healthy/'):
#     paths.append(os.path.join('./Healthy/',i))

# paths = sorted(paths, key=str.lower)

# model = Data_Aug_Model().to(device)

# # Load the state dict into the model
# model.load_state_dict(torch.load('./Pre_trained_Model/frame_interpolation_model.pth', map_location=torch.device('cpu')))

# for i in range(0, len(paths)-1):
#     frame = Image.open(paths[i])
#     fram = Image.open(paths[i+1])
#     frame1 = torch.tensor(np.array(frame), dtype=torch.float32).permute(2, 0, 1)/255.0 # [3, 200, 200]
#     frame2 = torch.tensor(np.array(fram), dtype=torch.float32).permute(2, 0, 1) /255.0
#     inputs = (torch.cat((frame1, frame2), 0).unsqueeze(0)).to(device)
#     op = model(inputs)
#     l = op[0].permute(2,1,0)
#     cp = l.detach().cpu().numpy()
#     maxi = np.max(cp)
#     cp = (cp/maxi)*255
#     data = Image.fromarray(cp.astype(np.uint8))
#     data.save(f"./results/image_{i+1}.png")
#     break
    

import os
import torch
import torch.nn as nn
from Frame_interpolation import Data_Aug_Model
from PIL import Image
import numpy as np

# Check if CUDA (GPU) is available, and set the device accordingly
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Create a list of file paths for the images in the 'Healthy' folder
paths = [os.path.join('./Healthy/', i) for i in os.listdir('./Healthy/')]
paths = sorted(paths, key=str.lower)

# Instantiate the Data_Aug_Model and move it to the selected device
model = Data_Aug_Model().to(device)

# Load the pre-trained model weights into the model with map_location=torch.device('cpu')
model.load_state_dict(torch.load('./Pre_trained_Model/frame_interpolation_model.pth', map_location=torch.device('cpu')))

# Loop through pairs of consecutive images and perform frame interpolation
for i in range(0, len(paths) - 1):
    # Open the current and next frames
    frame = Image.open(paths[i])
    next_frame = Image.open(paths[i + 1])

    # Convert frames to PyTorch tensors and normalize to the range [0, 1]
    frame1 = torch.tensor(np.array(frame), dtype=torch.float32).permute(2, 0, 1) / 255.0  # [3, 200, 200]
    frame2 = torch.tensor(np.array(next_frame), dtype=torch.float32).permute(2, 0, 1) / 255.0

    # Concatenate and unsqueeze to create the input tensor
    inputs = (torch.cat((frame1, frame2), 0).unsqueeze(0)).to(device)

    # Forward pass through the model to perform frame interpolation
    op = model(inputs)

    # Permute the output tensor dimensions for image visualization
    l = op[0].permute(2, 1, 0)

    # Convert the output tensor to a NumPy array
    cp = l.detach().cpu().numpy()

    # Normalize the pixel values to the range [0, 255]
    maxi = np.max(cp)
    cp = (cp / maxi) * 255

    # Create an image from the NumPy array and save it
    result_image = Image.fromarray(cp.astype(np.uint8))
    result_image.save(f"./results/Images/image_{i + 1}.png")