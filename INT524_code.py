import pandas as pd

#preparing the dataset
data = pd.read_csv("DATA-FINAL.csv",usecols=[0,1,2,4,5,6,8,9,10,11,12,13,20])
print(data.head())
print(data.isnull())