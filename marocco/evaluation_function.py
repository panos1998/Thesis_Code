from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
def function_evaluation(classifiers: List[Any], X: List[List], y: List,
names: List[str]):
    for classifier, name in zip(classifiers, names):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.25, random_state=3)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn/(tn+fp)
        prec = tp/(tp+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        rec= tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec)
        fpr,tpr,_=roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label =name)
        print(acc, spec, prec, rec, f1)
    plt.plot(np.arange(0.0,1.1,0.1),np.arange(0.0,1.1,0.1), linestyle="--", label='random')
    plt.vlines(0,0,1, colors='orange')
    plt.hlines(1,0,1, colors='orange', label='ideal')
    plt.legend(loc='lower right')
    plt.show()