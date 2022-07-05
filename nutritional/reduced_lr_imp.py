#%%
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.svm import NuSVC
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
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
imputer = IterativeImputer(min_value=[0 for _ in range(10)])
X_imputed = imputer.fit_transform(df.iloc[:,:-1])
diabe = df.iloc[:,-1]
df = pd.DataFrame(X_imputed, columns = df.columns[:-1])
df['diabe'] = diabe
print(df.isna().sum())
standarization = StandardScaler()
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
X_standarized = standarization.fit_transform(X)
corr = np.around(df.corr(),decimals=2)
sn.heatmap(corr, annot=True, cmap="BrBG")
plt.show()
pca = PCA()
pca.fit(X_standarized)
eye=pca.transform(np.identity(10))
eye = np.around(eye, decimals=3)
print(np.around(pca.explained_variance_ratio_,decimals=3))
coeffs =pd.DataFrame(eye,columns=[x for x in range(0,10)],index = df.columns[:-1])
coeffs.to_csv('principal_components_reduced_lr_imp.csv')
df.describe().to_csv('stats_reduced_lr_imp.csv')
#%%
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standarized)
clf = NuSVC(nu=0.3)
y_pred = cross_val_predict(clf, X_normalized, y, cv=4)
tn, fp, fn, tp= confusion_matrix(y_true=y, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
print()
#%%
# %%
xtr = ExtraTreesClassifier()
y_pred = cross_val_predict(clf, X_standarized, y, cv=4)
tn, fp, fn, tp= confusion_matrix(y_true=y, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
# %%
estimators = [('svm',clf),('rf', xtr)]
stacking = StackingClassifier(estimators, final_estimator=xtr, cv=4)
y_pred = cross_val_predict(stacking, X_standarized, y, cv=4)
tn, fp, fn, tp= confusion_matrix(y_true=y, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
# %%
