#%%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from concat_df import function_concat_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
colnames =['age', 'sex', 'cp', 'trestbps', 'chol',
 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope',
  'ca', 'thal', 'cvd']
source = ['processed.cleveland.data', 'processed.hungarian.csv',
'processed.va.csv', 'processed.switzerland.csv']
dest = ['label1.csv', 'label2.csv', 'label3.csv', 'label4.csv']
df = function_concat_df(dest=dest, labels=colnames, source=source)
df['cvd'] = df['cvd'].replace([2,3,4], 1) # replace cvd 2,3,4 with 1
print(df.isna().sum())
scaler = MinMaxScaler() # initialize a min max scaler
X = df[colnames[:-1]] # select features
y = df[colnames[-1]]  # select class
X_norm = scaler.fit_transform(X) # apply minmax to features
nb = GaussianNB()
svm = SVC()
scores = np.array(np.zeros((38,2)))
for i in range(1,39): # for different values impute with KNN
    imputer = KNNImputer(n_neighbors=i)
    X = imputer.fit_transform(X_norm)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.25, random_state =None)
    nb.fit(X_train, y_train)
    acc= nb.score(X_test, y_test)
    scores[i-1,0] = acc
    svm.fit(X_train, y_train)
    acc= svm.score(X_test, y_test)
    scores[i-1,1] = acc
plt.plot(range(1,39),scores[:,0],
label='Naive Bayes', marker='h')
plt.plot( range(1,39),scores[:,1],
label='SVM RBF', marker='h')
plt.title('Accuracy with respect to number of neighbors')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
