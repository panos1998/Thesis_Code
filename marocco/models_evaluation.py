#%%
import pandas as pd
from concat_df import function_concat_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
ExtraTreesClassifier, StackingClassifier)
from sklearn.svm import NuSVC
import lightgbm as lgb
from evaluation_function import  function_evaluation
colnames =['age', 'sex', 'cp', 'trestbps', 'chol',
 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope',
  'ca', 'thal', 'cvd']
source = ['processed.cleveland.data', 'processed.hungarian.csv',
'processed.va.csv', 'processed.switzerland.csv']
dest = ['label1.csv', 'label2.csv', 'label3.csv', 'label4.csv']
df = function_concat_df(dest=dest, labels=colnames, source=source)
df['cvd'] = df['cvd'].replace([2,3,4], 1) # replace cvd 2,3,4 with 1
df.to_csv('data.txt',index=False)
scaler = MinMaxScaler() # initialize a min max scaler
X = df[colnames[:-1]] # select features
y = df[colnames[-1]]  # select class
X_norm = scaler.fit_transform(X) # apply minmax to features
X_norm = pd.DataFrame(X_norm, columns=colnames[:-1])
imputer = IterativeImputer(imputation_order='roman', sample_posterior=False,
max_iter=40, random_state=29, add_indicator=False)
X_lr = imputer.fit_transform(X_norm)[:,0:13]
names = ['XGradientBoosting','Adaboost','GradientBoosting','ExtraTrees',
'LGBM','SGDC', 'Nu-SVC', 'Stacking']
xgb = XGBClassifier(n_estimators =100,learning_rate=0.1)
#0.8608695652173913 0.8369565217391305 0.8897058823529411 0.8768115942028986 0.8832116788321168
adb= AdaBoostClassifier(learning_rate=1, n_estimators=50)
gdb = GradientBoostingClassifier(learning_rate=1, n_estimators=3)
xtr = ExtraTreesClassifier(n_estimators=80)
lgbm = lgb.LGBMClassifier(learning_rate=0.009,n_estimators=1000, objective='binary')
sgdc = SGDClassifier(learning_rate='adaptive',loss='log', eta0=1)
nsvc = NuSVC(nu = 0.25)
stck = StackingClassifier(estimators=[('xgb',xgb),('adb',adb), ('xtr',xtr),
('lgbm',lgbm), ('sgdc',sgdc),('nsvc', nsvc), ('gdb',gdb)], final_estimator=nsvc)
classifiers = [xgb,adb,gdb,xtr,lgbm,sgdc,nsvc,stck]
function_evaluation(classifiers, X_lr,y, names)
# %%
