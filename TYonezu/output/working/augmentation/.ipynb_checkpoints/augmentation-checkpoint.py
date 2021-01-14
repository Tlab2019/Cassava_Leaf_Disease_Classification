import numpy as np 
from torch import nn
import pandas as pd

def EqualizeLabels(train_df,NUM):
    train_df = train_df[["image_path","label"]]
    out = pd.concat( [df.sample(n=NUM, replace=True) for l,df in train_df.groupby("label")] )

    return out