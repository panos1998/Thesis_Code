#%%
from typing import List
import pandas as pd
from concat_df import function_concat_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
X_train, X_test, y_train, y_test = train_test_split(X_mean, y,
test_size = 0.25, random_state =6)
X_knn= imputer.fit_transform(X_norm)
X_k_train, X_k_test, y_k_train, y_k_test = train_test_split(X_knn, y,
test_size = 0.25, random_state =6)
imputer = IterativeImputer(estimator=RandomForestRegressor(),max_iter=10, random_state=6)
X_rf = imputer.fit_transform(X_norm)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y,
test_size = 0.25, random_state =6)
imputer = IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=6)
X_lr = imputer.fit_transform(X_norm)
X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y,
test_size = 0.25, random_state =6)
# classifiers
linear = SVC(kernel='linear', probability=True)
rbf = SVC(probability=True)
nb = GaussianNB()
ann = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,100))
clfs = [linear, rbf, nb, ann]
features = [X_mean, X_knn, X_rf, X_lr]
names = ['linear  SVM', 'rbf SVM', 'Naive Bayes', 'ANN']
for clf, name in zip(clfs, names):
  function_clf_by_data(clf,features=features,y=y, name=name)
# %%
