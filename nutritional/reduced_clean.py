#%%
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
df = pd.read_csv(filepath_or_buffer='test.csv', usecols=range(20))
df['diabe'] = [0 for _ in range(119)]
df['diabe'] = np.where((df['FPG (mg/dL)']>126) & (df['Diastolic BP (mmHg)']>85) & (df['Systolic BP (mmHg)']>135), 1, 0)
df.to_csv('dataset_with_class.csv',index_label=False, index=False)
#
print(df.groupby('diabe').count())

#%%
#df.drop(['HDL-c (mg/dl)', 'LDL-c (mg/dl)', 'Total Cholesterol (mg/dL)',
#'Atherogenenciity index (Total/ HDL)', 'TAG (mg/dL)',
#'FPG (mg/dL)', 'HbA1c (%)', 'CRP (mg/L)'], axis=1, inplace=True)
df.drop(['FPG (mg/dL)', 'Diastolic BP (mmHg)', 'Systolic BP (mmHg)', 'HDL-c (mg/dl)',
'LDL-c (mg/dl)', 'Total Cholesterol (mg/dL)', 'Atherogenenciity index (Total/ HDL)',
'TAG (mg/dL)','HbA1c (%)', 'CRP (mg/L)'],
inplace=True, axis=1)
print(df.head())
#df.to_csv('non_invasive.csv',index_label=False, index=False)
#%%
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
standarization = StandardScaler()
X_clean = X.dropna()

X_standarized = standarization.fit_transform(X_clean)
corr = np.around(X_clean.corr(),decimals=2)
#f, ax = plt.subplots(figsize=(30, 15))
#f.set_figwidth(20)
#f.set_figheight(20)
sn.heatmap(corr, annot=True, cmap="BrBG")
#f.set_tight_layout(True)
plt.show()
pca = PCA()
pca.fit(X_standarized)
eye=pca.transform(np.identity(10))
eye = np.around(eye, decimals=3)
print(np.around(pca.explained_variance_ratio_,decimals=3))
coeffs =pd.DataFrame(eye,columns=[x for x in range(0,10)],index = df.columns[:-1])
coeffs.to_csv('principal_components_reduced.csv')
# %%
