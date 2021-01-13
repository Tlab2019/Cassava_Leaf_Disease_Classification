import numpy as np 
from torch import nn

def NN_Trainer(model,optimizer,criterion,device,accuracy,epoch):

    model.train()
    
    for epoch in tqdm(range(EPOCH_NUM)):
        loss = 0

        for batch in (train_data):

            X = batch[0].cuda()
            y = batch[1].cuda()

            pred = model(X)

            # zero the gradient buffers
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()

            loss += loss.item()
            optimizer.step() # Does the update

        print("epoch:",epoch,"|loss:",loss)
        
