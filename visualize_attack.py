import torch
import pickle
import os
from torchvision import transforms

# 强制指定当前目录
current_dir = "."
output_dir = "./visualized_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tp = transforms.ToPILImage()
files = [f for f in os.listdir(current_dir) if f.endswith('.bin')]

print(f"Found {len(files)} files. Starting...")

for f_name in files:
    try:
        with open(f_name, "rb") as f:
            data = pickle.load(f)

        # 转换并归一化
        tensor = data.detach().cpu().squeeze(0)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

        img = tp(tensor)
        img.save(os.path.join(output_dir, f_name.replace(".bin", ".png")))
    except Exception as e:
        print(f"Error processing {f_name}: {e}")

print("Done! Please check 'visualized_images' folder.")