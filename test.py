from PIL import Image
import os, shutil

original = sorted(os.listdir("./Healthy"),key=str.lower)
# generated = sorted(os.listdir("./results/Images"), key=str.lower)

for i in range(0, len(original)-1):
    shutil.copy(f"./Healthy/image_{str(i + 1).zfill(3)}.png", f"./Interpolated/image_{i * 2}.png")
    # shutil.copy(f"./Healthy/{original[i + 1]}", f"./Interpolated/image_{i + 2}.png")
    shutil.copy(f"./results/Images/image_{i + 1}.png", f"./Interpolated/image_{i * 2 - 1}.png")