import pandas as pd

#preparing the dataset
initial_dataset = pd.read_csv("DATA-FINAL.csv",usecols=[0,1,2,4,5,6,8,9,10,11,12,13,20])
print(dataset.head())
working_dataframe = initial_dataset[initial_dataset["MHRDName"= Bachelor of Science (Honours) (Agriculture)],initial_dataset["CourseType"=Theory]]
print(working_dataframe.head())

#choosing the model