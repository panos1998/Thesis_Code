#%%
from arfftocsv import arfftocsv, labelize, dataEncoding
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
stats = []
aucs = np.zeros((10, 8, 100))
mean_auc_per_epoch = 0
mean_aucs = np.array(np.zeros((100,10)), dtype=float)
#for j in range (500):         # TRUE POSITIVE RATE = SENSITIVITY
for k in tqdm.tqdm(range(100), colour='CYAN'):
    for i in range(0,10):
       j = 0                                       # TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y, 
       train_size=  0.7, test_size=0.3, stratify=y)
       for c in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
           clf = LogisticRegression(C=c, solver='liblinear', penalty='l2').fit(X_train, y_train)
           y_pred = (clf.predict_proba(X_test))[:,1]
    #print(classification_report(y_test, y_pred))
    #print(recall_score(y_test, y_pred, average=None))
    #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred)
           stats.append([1-fpr, sensitivity])
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred))
           j = j + 1
means = np.mean(np.mean(aucs, axis =2), axis =0)
print(means)
plt1 = plt.figure(1)
plt.plot(['0.0001', '0.001', '0.01', '0.1', '1', '10', '100', '1000'], means)
   # mean_auc_per_epoch = np.mean(aucs, axis = 0)
    #mean_aucs[k] = mean_auc_per_epoch
#print(np.mean(mean_aucs, axis = 0))

# να δουμε αυριο τι να αλλαξουμε μπας κ φτασουμε τα σωστα, οπως regularization C, penalty l1, l2 
#Υπαρχει μια απόκλιση 4%
#Εδω θα βαλουμε αυριο το κωδικα για να βρουμε το threshold, να ψαξω αν πρωτα πρεπει να κανω  training  και μετα hyper
# parameters ή βολευει οπως το εκανα
 
scores = list()
thresholds = np.linspace(0, 1, 1000)
for thr in tqdm.tqdm(thresholds):
    younden = list()
    for i in range(0,10):                                      # TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y, 
       train_size=  0.7, test_size=0.3, stratify=y)
       clf = LogisticRegression(C=100, solver='liblinear', penalty='l2').fit(X_train, y_train)
       y_pred = (clf.predict_proba(X_test)[:,1] >= thr).astype(bool)
       tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
       specificity = tn/(tn+fp)
       sensitivity = tp/(tp+fn)
       younden.append((specificity+sensitivity-1,specificity, sensitivity))
    scores.append((sum(you[0] for you in younden)/ len(younden),
    sum(you[1] for you in younden)/ len(younden),
    sum(you[2] for you in younden)/ len(younden), thr))
params =list(clf.get_params().values()) 
optimal =max(scores, key=lambda score: score[0])
print('Maximum younden,specificity, sensitivity, threshold, c and penalty ',optimal,params[0], params[9])
plt2 = plt.figure(2)
plt.plot([x[3] for x in scores], [x[0] for x in scores])
plt.show()

# l1 c = 10, 0.377 για  τα 10 μεγαλα dataset
#ωρα για τα 10 μικρα l1 c = 10 0.4279 0.2020 ρεαλιστικα 0.41 με 0.151
# l2 c = 1000 0.445 0.161 realistika 0.412 thr 0.151, y 0.44 sp 0.62, sens 0.82 thr 0.134
# τελικη επιλογη l2 γιατι ειναι αρκετα γρηγοροτερη και c = 100


### FINAL EVALUATION ######

younden_final = list()
for i in tqdm.tqdm(range(0,10)):                                      # TRUE NEGATIVE RATE = SPECIFICITY
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
    train_size=  0.7, test_size=0.3, stratify=y)
    clf = LogisticRegression(C=100, solver='liblinear', penalty='l2').fit(X_train, y_train)
    y_pred = (clf.predict_proba(X_test)[:,1] >= optimal[3]).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn/(tn+fp)
    sensitivity = tp/(tp+fn)
    younden_final.append((specificity+sensitivity-1,specificity, sensitivity))
younden_final = np.array(younden_final)
fig, ax = plt.subplots(3, 1)
fig.tight_layout()
ax[0].plot(younden_final[:,0])
ax[1].plot(younden_final[:,1])
ax[2].plot(younden_final[:,2])
ax[0].set_ylabel('Younden Index')
ax[2].set_ylabel('Sensitivity')
ax[1].set_ylabel('Specificity')
plt.show()
print(np.mean(younden_final, axis=0))
# %%
