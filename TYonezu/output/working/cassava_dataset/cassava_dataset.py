import numpy as np 
from PIL import Image
import torch
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'test': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
    
    
class CassavaDataset(data.Dataset):
    def __init__(self, file_dict, transform=None, phase='train'):
        self.file_path = list(file_dict.keys())
        self.file_dict = file_dict
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, index):
        img_path = self.file_path[index]
        img = Image.open(img_path)
        img = self.transform(img, self.phase)
        
        label = self.file_dict[img_path]
        
        return img, label