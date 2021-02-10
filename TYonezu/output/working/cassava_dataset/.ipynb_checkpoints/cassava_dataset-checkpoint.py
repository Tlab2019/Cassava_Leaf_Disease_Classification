import numpy as np 
from PIL import Image
import torch
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(resize),
                transforms.RandomRotation(degrees=(-180,180)),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            
            'val': transforms.Compose([
                transforms.Resize(resize),
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
    
def cassava_predict_label(model,test_dataloader,sub_df,device="cuda"):
    
    cols = ["pred_label"]
    
    model.eval()
    model = model.to(device)
    outputs = []

    for batch in tqdm(test_dataloader):

        X = batch[0].to(device)
        y = batch[1].to(device)
        
        pred = model(X)
        pred = torch.nn.Softmax(dim=1)(pred)
        pred = pred.cpu().detach().numpy()
        pred = pred.argmax(axis=1)
        
        pred = pd.DataFrame(columns=cols,
                             data=pred)

        outputs.append(pred)

    outputs = ( pd.concat(outputs) ).reset_index(drop=True)
    sub_df = pd.concat([sub_df,outputs],axis=1)
    
    return sub_df[["image_id"]+cols]


def cassava_predict_proba(model,test_dataloader,sub_df,device="cuda"):
    
    cols = ["class0_proba",
            "class1_proba",
            "class2_proba",
            "class3_proba",
            "class4_proba"]
    
    model.eval()
    model = model.to(device)
    outputs = []

    for batch in tqdm(test_dataloader):

        X = batch[0].to(device)
        y = batch[1].to(device)
        
        pred = model(X)
        pred = torch.nn.Softmax(dim=1)(pred)
        pred = pred.cpu().detach().numpy()
        
        pred = pd.DataFrame(columns=cols,
                             data=pred)

        outputs.append(pred)

    outputs = ( pd.concat(outputs) ).reset_index(drop=True)
    sub_df = pd.concat([sub_df,outputs],axis=1)
    
    return sub_df[["image_id"]+cols]
