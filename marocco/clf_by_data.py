from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
def function_clf_by_data(clf, features: List, name:str,
y: pd.DataFrame(),labels:List[str]=['mean', 'knn', 'rf','mice'],
colors:List[str]= ['red', 'yellow', 'green', '#1f77b4']):
  for X, color, label in zip(features, colors, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size = 0.25, random_state =100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr, label=label, color =color)
  plt.title(name)
  plt.legend()
  plt.show()