from torchvision import transforms
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as transF


def imshow(images, title=None):
    images = images.numpy().transpose((1, 2, 0))  # (h, w, c)
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #images = std * images + mean
    #images = np.clip(images, 0, 1)
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    
class TestTimeAugmentation(object):
    
    def __init__(self,model,size=(224,224)):
        self.model = model
        self.size = size
    
    
    def predict(self,batch_input,device = "cuda"):
        # batch_input => (batch_size,channel,W,H)
        
        batch_size = batch_input.shape[0]
        
        output = []
        self.model = self.model.to(device)
        self.model.eval()
        for b in range(batch_size):
            
            original_img = batch_input[b]
            pil_img = transF.to_pil_image(original_img)
            
            imgs = []
            imgs.append( transF.to_tensor(pil_img) ) # original image
            imgs.append( transF.to_tensor(transF.rotate(pil_img, angle=90)) ) #rotate 90
            imgs.append( transF.to_tensor(transF.rotate(pil_img, angle=270)) ) #rotate 270
            imgs.append( transF.to_tensor(transF.vflip(pil_img)) ) # vertical flip
            imgs.append( transF.to_tensor(transF.hflip(pil_img)) ) # horizontal flip
            
            imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            pred = self.model(imgs)
            pred = (torch.nn.Softmax(dim=1))(pred)
            
            print(pred.argmax(axis=1))
            
            pred = pred.mean(axis=0)
            
            
            output.append(pred)
            
        output = torch.stack(output)
        
        return output