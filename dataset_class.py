# Dataset can be downloaded from https://github.com/ayulockin/synthetic_datasets/tree/master
import pandas as pd
import numpy as np
import os 
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image

path = r"C:\Users\matan\Desktop\MLDL_Projects\MNIST_Localization\Data\training_data.csv"
df = pd.read_csv(path, header=None)

class MNIST(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        name = self.df.iloc[index, 0]
        target_lis = self.df.iloc[index, 1:6].tolist()
        label = torch.tensor(target_lis[0] , dtype=torch.long)
        bbox = torch.tensor(target_lis[1:] , dtype=torch.float32)

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return (img, {'label': label , 'bbox': bbox})
    

# https://kozodoi.me/blog/20210308/compute-image-stats#3.-Computing-image-stats    #computation time took 9m 3.4s on CPU
if __name__ == "__main__":

    img_dir = r"C:\Users\matan\Desktop\MLDL_Projects\MNIST_Localization\Data\MNIST_Converted_Training\\"
    csv_file = r"C:\Users\matan\Desktop\MLDL_Projects\MNIST_Localization\Data\training_data.csv"

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])    # Resizing for resnet model

    train_ds = MNIST(csv_file=csv_file , img_dir=img_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    sum = torch.zeros(3)
    sum_sq = torch.zeros(3)
    count = len(train_ds)*224*224

    for imgs, _ in train_loader:
        sum += imgs.sum(dim= [0,2,3])
        sum_sq += (imgs**2).sum(dim= [0,2,3])

    total_mean = sum/count
    total_var = (sum_sq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print(f'mean is: {str(total_mean)}')    # mean is: tensor([0.0102, 0.0102, 0.0102])
    print()
    print(f'std is: {str(total_std)}')      # std is: tensor([0.0882, 0.0882, 0.0882])