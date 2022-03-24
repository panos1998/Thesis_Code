from pickle import TRUE
from arfftocsv import arfftocsv, labelize, dataEncoding
from operator import index
import pandas as pd
import numpy as np
import tqdm
all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm','mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge',
'hearte','psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m','jphysa','estwt','wstva','chol','hdl','ldl',
'trig','sys1','dias3','fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']

labels = ['LeicAge','LeicGender','bmicat','LeicRace','hemda','rYdiabe']

to_replace = {'LeicAge': ['50-59', '60-69', '>=70'], 'LeicGender': ['Female', 'Male'], 
'bmicat': ["'1.underweight less than 18.5'", "'2.normal weight from 18.5 to 24.9'", "'3.pre-obesity from 25 to 29.9'",
"'4.obesity class 1 from 30 to 34.9'", "'5.obesity class 2 from 35 to 39.9'", "'6.obesity class 3 greater than 40'"],
'LeicRace': [0, 6], 'hemda': ["'Not applicable'", 'Yes', 'No'], 'rYdiabe': ['0.no', '1.yes']
}

values = {'LeicAge': [0, 1, 2], 'LeicGender': [0, 1], 'bmicat': [1, 2, 3, 4, 5, 6], 'LeicRace': [0, 6],
 'hemda': [0, 1, 2], 'rYdiabe': [0, 1]}

def processing (all_labels: list, labels: list, to_replace: dict, values: dict, path, source: str = 'diabetes_paper_fazakis.csv',
 des: str  ='Finaldata.csv')-> pd.DataFrame:
 arfftocsv(source)
 df = labelize(des, all_labels)
 return dataEncoding(df, labels, to_replace, values, path)
path = 'LeicLog.csv'
data = processing(all_labels, labels, to_replace, values, path)

# Apply machine learning techniques
X = data[labels[:-1]]
y = data[labels[len(labels)-1]]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
stats = []
thresholds = np.linspace(0, 1, 100)
for thr in tqdm.tqdm(thresholds, colour='CYAN'):
    for i in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        clf = LogisticRegression().fit(X_train, y_train)
        y_pred = (clf.predict_proba(X_test)[:,1]>= thr).astype(bool)
        #print(classification_report(y_test, y_pred))
        #print(recall_score(y_test, y_pred, average=None))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        stats.append([tp/(tp + fn),tn/(tn + fp), roc_auc_score(y_test, y_pred), thr])
        #print('Mean sensitivity', 'Mean specificity', 'AUC',' Threshold', np.mean(stats, axis=0), thr)
best = max(stats, key= lambda x:x[0] +x[1] - 1)
print('Younden Index', best[0] + best[1] - 1, best)
    #print(np.mean(stats[:,0]), np.mean(stats[:,1]))
    # να δουμε αυριο τι να αλλαξουμε μπας κ φτασουμε τα σωστα, οπως regularization C, penalty l1, l2 