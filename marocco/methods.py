#%%
from typing import List
import pandas as pd
from sklearn.tree import ExtraTreeRegressor
from concat_df import function_concat_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPClassifier
from clf_by_data import function_clf_by_data
#%%
imputer = KNNImputer(n_neighbors=6)
colnames =['age', 'sex', 'cp', 'trestbps', 'chol',
 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope',
  'ca', 'thal', 'cvd']
source = ['processed.cleveland.data', 'processed.hungarian.csv',
'processed.va.csv', 'processed.switzerland.csv']
dest = ['label1.csv', 'label2.csv', 'label3.csv', 'label4.csv']
df = function_concat_df(dest=dest, labels=colnames, source=source)

df['cvd'] = df['cvd'].replace([2,3,4], 1) # replace cvd 2,3,4 with 1
scaler = MinMaxScaler() # initialize a min max scaler
X = df[colnames[:-1]] # select features
y = df[colnames[-1]]  # select class
X_norm = scaler.fit_transform(X) # apply minmax to features
X_norm = pd.DataFrame(X_norm, columns=colnames[:-1])
X_mean=X_norm.fillna(X_norm.mean(), inplace=False)
X_knn= imputer.fit_transform(X_norm) # X with NN
imputer = IterativeImputer(estimator=ExtraTreeRegressor(),max_iter=100, random_state=29,
imputation_order='random')
X_rf = imputer.fit_transform(X_norm)
imputer = IterativeImputer(imputation_order='random',max_iter=100, random_state=29)
X_lr = imputer.fit_transform(X_norm)

# classifiers
linear = SVC(kernel='linear')
rbf = SVC()
nb = GaussianNB()
ann = MLPClassifier()
clfs = [linear, rbf, nb, ann]
features = [X_mean, X_knn, X_rf, X_lr]
names = ['Linear  SVM', 'RBF SVM', 'Naive Bayes', 'ANN']
for clf, name in zip(clfs, names): # for all classifiers evaluate rocs
  function_clf_by_data(clf,features=features,y=y, name=name)
# %%