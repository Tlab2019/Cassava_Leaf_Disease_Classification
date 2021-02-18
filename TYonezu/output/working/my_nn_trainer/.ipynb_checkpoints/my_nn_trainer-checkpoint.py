import numpy as np 
from torch import nn
import os
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

##############################################################
# A NeuralNet trainner pytroch implemantation
# pytorch version 1.7.0
###############################################################

def run_training(model,optimizer,criterion,epoch_num,
                 train_dataloader,valid_dataloader,MODEL_NAME,
                 device=None,verbose=-1):
    
    model = model.to(device)
    
    log = {}
    best_acc = -np.inf
    PATH = None

    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    
    pbar = tqdm(total=epoch_num)
    for epoch in range(epoch_num):
        
        print("===========================================")
        print("epoch: ",epoch,"\n")

        model.train()
        train_acc = 0
        train_loss = 0
        for batch in (train_dataloader):
            X = batch[0].to(device)
            y = batch[1].to(device)

            # zero the gradient buffers
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer) # Does the update
                scaler.update()

            pred_label = ( pred.argmax(axis=1) ).cpu().numpy()
            y = y.cpu().numpy()
            train_acc += (pred_label == y).sum()
            train_loss += loss.item() * X.size(0)

        train_acc = train_acc/len(train_dataloader.dataset)
        train_loss = train_loss/len(train_dataloader.dataset)


        model.eval()
        valid_acc = 0
        valid_loss = 0
        for batch in (valid_dataloader): 
            X = batch[0].to(device)
            y = batch[1].to(device)

            with torch.cuda.amp.autocast():
                pred = model(X)
                loss = criterion(pred, y)

            pred_label = ( pred.argmax(axis=1) ).cpu().numpy()
            y = y.cpu().numpy()

            valid_acc += (pred_label == y).sum()
            valid_loss += loss.item() * X.size(0)

        valid_acc = valid_acc/len(valid_dataloader.dataset)
        valid_loss = valid_loss/len(valid_dataloader.dataset)
        
        print(" > train loss:",train_loss,"  valid loss:",valid_loss)
        print(" > train acc:",train_acc,"  valid acc:",valid_acc)
        print(" > best acc: ",best_acc)

        if valid_acc > best_acc:
            print("   ! best acc updated ! => ",valid_acc)
            
            if PATH is not None:
                os.remove(PATH) # remove old best model
            
            PATH = MODEL_NAME+"-epoch%d"%epoch+".pth"
            torch.save(model.state_dict(),PATH)
            best_acc = valid_acc        
        
        log[epoch] = [train_acc, train_loss, valid_acc, valid_loss]

        if (verbose>0)&(epoch%verbose == 0):
            log_df = pd.DataFrame.from_dict(log,
                                            orient="index",
                                            columns=["train acc","train loss","valid acc","valid loss"])

            fig = plt.figure(figsize=(10,3))

            ax = fig.add_subplot(1,2,1)
            log_df[["train loss","valid loss"]].plot(marker="o",ax=ax)
            ax.set_xlim(0,epoch_num+1)
            ax.grid(True)

            ax2 = fig.add_subplot(1,2,2)
            log_df[["train acc","valid acc"]].plot(marker="o",ax=ax2)
            ax2.grid(True)
            ax2.set_xlim(0,epoch_num+1)

            fig.suptitle(MODEL_NAME+"-epoch%d"%epoch)
            plt.savefig(MODEL_NAME+".jpg",format="jpg")
            plt.show()
        
        # update progress bar
        pbar.update(1)
    
    pbar.close()
    
    return pd.DataFrame.from_dict(log,orient="index",columns=["train acc","train loss","valid acc","valid loss"])