from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np 

trans4tta = [
    transforms.RandomRotation(degrees=(0,0),expand=True),
    transforms.RandomRotation(degrees=(90,90),expand=True),
    transforms.RandomRotation(degrees=(270,270),expand=True),
    transforms.RandomHorizontalFlip(p=1),
    #transforms.RandomVerticalFlip(p=1),
]

for i,trans in enumerate(trans4tta):
    trans4tta[i] = transforms.Compose([
        transforms.ToPILImage(),
        trans,
        transforms.ToTensor()
    ])

class TestTimeAugmentation(object):
    
    def __init__(self,model,device,trans4tta=trans4tta):
        self.model = model.to(device)
        self.device = device
        self.trans4tta = trans4tta
    
    
    def predict(self,batch_input):
        # batch_input => (batch_size,channel,W,H)
        
        batch_size = batch_input.shape[0]
        
        output = []
        for b in range(batch_size):
            
            original_img = batch_input[b]
            
            aug_imgs = []
            for trans in self.trans4tta:
                
                tmp = trans(original_img)
                aug_imgs.append(tmp)
                
                #plt.imshow(tmp[0])
                #plt.show()
            
            aug_imgs = torch.stack(aug_imgs)
            aug_imgs = aug_imgs.to(self.device)
            
            pred = self.model(aug_imgs)
            pred = (torch.nn.Softmax(dim=1))(pred)
            pred = pred.mean(axis=0)
            
            output.append(pred)
            #print(pred.argmax())
            
        output = torch.stack(output)
        
        return output