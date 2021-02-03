import numpy as np 
from torch import nn
import os
import torch

##############################################################
# A pytorch implementation of https://arxiv.org/abs/1409.1556 
###############################################################

class NN_TRAINER(object):
    
    def __init__(self,model,criterion,optimizer,OUT_DIR,MODEL_NAME):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.best_acc = -np.inf
        
        self.epoch = 0
        self.log = {}
        
        self.MODEL_NAME = MODEL_NAME
        self.PATH = os.path.join(OUT_DIR,MODEL_NAME)
        
        if not(os.path.exists(OUT_DIR)):
            os.makedirs(OUT_DIRS)
    
    def run_one_epoch(self,train_dataloader,valid_dataloader,device="cpu"):
        
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        
        self.model = self.model.to(device)
        self.model.train()
        train_acc = 0
        train_loss = 0
        for batch in (train_dataloader):
            X = batch[0].to(device)
            y = batch[1].to(device)

            # zero the gradient buffers
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = self.model(X)
                loss = self.criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer) # Does the update
                scaler.update()

            pred_label = ( pred.argmax(axis=1) ).cpu().numpy()
            y = y.cpu().numpy()
            train_acc += (pred_label == y).sum()
            train_loss += loss.item() * X.size(0)

        train_acc = train_acc/len(train_data.dataset)
        train_loss = train_loss/len(train_data.dataset)
        
        model.eval()
        valid_acc = 0
        valid_loss = 0
        for batch in (valid_data): 
            X = batch[0].to(device)
            y = batch[1].to(device)

            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)

            pred_label = ( pred.argmax(axis=1) ).cpu().numpy()
            y = y.cpu().numpy()

            valid_acc += (pred_label == y).sum()
            valid_loss += loss.item() * X.size(0)

        valid_acc = valid_acc/len(valid_data.dataset)
        valid_loss = valid_loss/len(valid_data.dataset)
        
        
        self.epoch += 1
        self.log[self.epoch] = [train_acc, train_loss, valid_acc, valid_loss]
        
        return [train_acc, train_loss, valid_acc, valid_loss]
    
    
    def plot_log(self):
        log_df = pd.DataFrame.from_dict(self.log,
                                        orient="index",
                                        columns=["train acc","train loss","valid acc","valid loss"])
        
        fig = plt.figure(figsize=(10,3))
        
        ax = fig.add_subplot(1,2,1)
        log_df[["train loss","valid loss"]].plot(marker="o",ax=ax)
        ax.grid(True)
        
        ax2 = fig.add_subplot(1,2,2)
        log_df[["train acc","valid acc"]].plot(marker="o",ax=ax2)
        ax2.grid(True)
        
        fig.suptitle(self.MODEL_NAME)
        plt.savefig(MODEL_NAME+".png",format="png")
        plt.show()
    
    def run(self,train_dataloader,valid_dataloader,epoch_num,device="cpu"):
        
        
        
        for i in range(epoch_num):
            result = self.run_one_epoch(train_dataloader,
                                        valid_dataloader,
                                        device=device)
            
            self.plot_log()
            
            train_acc, train_loss, valid_acc, valid_loss = result
            
            if valid_acc > self.best_acc:
                torch.save(self.model.state_dict(), self.PATH)
                self.best_acc = valid_acc