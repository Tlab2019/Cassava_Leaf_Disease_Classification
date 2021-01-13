import numpy as np 
from torch import nn

def MyAugumentation(train_df):
    
    train_df = train_df[["image_path","label"]]
    
    