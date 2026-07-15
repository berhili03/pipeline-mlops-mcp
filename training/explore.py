import pandas as pd 

df = pd.read_csv("data/telco_churn.csv")

#print(df.shape)
#print(df.columns)
#print(df.head())
#print(df.dtypes)
print(df.isnull().sum())