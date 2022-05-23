import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
lm=LinearRegression()

#reading the dataset
path='no_null_df.csv'
df = pd.read_csv(path)

print(df['Make'].nunique())
# cols=df.columns
# for c in cols:
#     if(df[c].dtype==object):
#         if(df[c].nunique()>10 and df[c].nunique()<20):
#             print(c)