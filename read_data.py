import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

class HRDataset(Dataset):
    def __init__(self, hr_dir):
        self.files = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert("RGB")
        return hr
    

class BicubicDownsample(Dataset):
    def __init__(self, hr_dataset, scale=4):
        self.hr_dataset = hr_dataset
        self.scale = scale

    def __len__(self):
        return len(self.hr_dataset)

    def __getitem__(self, idx):
        hr = self.hr_dataset[idx]
        lr = hr.resize(
            (hr.size[0]//self.scale, hr.size[1]//self.scale),
            Image.BICUBIC
        )
        lr_up = lr.resize(hr.size, Image.BICUBIC)

        lr_t = TF.to_tensor(lr_up)
        hr_t = TF.to_tensor(hr)

        return lr_t, hr_t
    
def plot_lr_hr_comparison(dataset, num_samples=4, figsize=(15, 8)):

    num_samples = min(num_samples, len(dataset))
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        lr_t, hr_t = dataset[i]

        lr_img = lr_t.permute(1, 2, 0).numpy()
        hr_img = hr_t.permute(1, 2, 0).numpy()

        axes[0, i].imshow(lr_img)
        axes[0, i].set_title(f'LR Upsampled #{i+1}\n{lr_img.shape[1]}x{lr_img.shape[0]}')
        axes[0, i].axis('off')

        axes[1, i].imshow(hr_img)
        axes[1, i].set_title(f'HR Original #{i+1}\n{hr_img.shape[1]}x{hr_img.shape[0]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()