#%%
import sys
import numpy as np
from typing import Any, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sys.path.append('C:/Users/panos/Documents/Διπλωματική/code/fz')
from arfftocsv import function_labelize
import csv
colnames =['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
'exang', 'oldpeak', 'slope', 'ca', 'thal', 'cvd']
# %%
df1 = function_labelize(dest = 'labeled_data1.txt', labels=colnames, source = 'processed.hungarian.csv')
df2 = function_labelize(dest = 'labeled_data2.txt', labels=colnames, source = 'processed.cleveland.data')
df3 = function_labelize(dest = 'labeled_data3.txt', labels=colnames, source = 'processed.va.csv')
df4 =function_labelize(dest = 'labeled_data4.txt', labels=colnames, source = 'processed.switzerland.csv')
df = pd.concat([df1,df2,df3,df4], axis=0)
print(df.isna().sum())
df['cvd'] = df['cvd'].replace([2,3,4], 1)
scaler = MinMaxScaler()
X = df[colnames[:-1]]
y = df[colnames[-1]]
X_norm = scaler.fit_transform(X)
print(X_norm)
print(y)
# %%
