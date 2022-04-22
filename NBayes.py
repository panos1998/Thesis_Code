#%%
from turtle import width

from sklearn import neighbors
from arfftocsv import processing
import numpy as np
import pandas as pd
import tqdm
all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm',
'mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge','hearte',
'psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m',
'jphysa','estwt','wstva','chol','hdl','ldl','trig','sys1','dias3',
'fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']
# this function deletes @ and empty lines so that produce a 

to_replace = {'LeicAge': ['50-59', '60-69', '>=70'], 'LeicGender': ['Female', 'Male'], 
'bmicat': ["'1.underweight less than 18.5'",
 "'2.normal weight from 18.5 to 24.9'", "'3.pre-obesity from 25 to 29.9'",
 "'4.obesity class 1 from 30 to 34.9'", "'5.obesity class 2 from 35 to 39.9'",
  "'6.obesity class 3 greater than 40'"],
'LeicRace': [0, 6], 'hemda': ["'Not applicable'", 'Yes', 'No'],
 'wstva':[],'LeicHBP': ['No', 'Yes'], 'rYdiabe': ['0.no', '1.yes'],
 'raeducl':["'2.upper secondary and vocational training'",'3.tertiary',
 "'1.less than secondary'",'.o:other',"'.h:missing HSE value'",".m:Missing"],
  'mstat':['3.partnered','1.married','5.divorced','7.widowed','4.separated',"'8.never married'"],
  'shlt':["'2.Very good'",'3.Good','1.Excellent','4.Fair','5.Poor'], 'hlthlm':['0.no','1.yes','.d:DK'],
  'hearte':['0.no','1.yes'],'psyche':['0.no','1.yes'], 'physActive':['Yes','No'], 'smoken':['0.No',
  '1.Yes',".m:Missing"],
   'jphysa': ["'1.Sedentary occupation'","'3.Physical work'","'2.Standing occupation'","'.w:not working'",
   "'4.Heavy manual work'",'.m:Missing'], 'everHighGlu':['No',"'Not applicable'",'Yes'],
   'eatVegFru':['Yes', 'No'],'mobilb':[],'lgmusa':[], 'grossa':[],'finea':[], 'drinkd_e':[],'itot':[],'cfoodo1m':
   [], 'estwt':[], 'chol':[],'ldl':[], 'hdl':[], 'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
}

values = {'LeicAge': [0, 1, 2], 'LeicGender': [0, 1],
 'bmicat': [1, 2, 3, 4, 5, 6],'LeicRace': [0, 1],
 'hemda': [0, 1, 2], 'wstva':[],'LeicHBP':[0,1],'rYdiabe': [0, 1], 'raeducl':[2,3,1,0,4,np.nan], 'mstat':[
     3,1,5,7,4,8],'shlt':[2,3,1,4,5], 'hlthlm':[0,1,2],'hearte':[0,1], 'psyche':[0,1], 'physActive':[1,0],
  'smoken':[0,1,np.nan], 'jphysa':[1,3,2,5,4,np.nan], 'everHighGlu':[0,2,1], 'eatVegFru':[1,0],'mobilb':[],'lgmusa':[],
  'grossa':[], 'finea':[], 'drinkd_e':[],'itot':[],'cfoodo1m':[], 'estwt':[],'chol':[],'ldl':[], 'hdl':[],
  'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
 }

path = 'NBayes.csv'
data = processing(labels=all_labels, to_replace=to_replace,all_labels=all_labels, values= values, path=path)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import roc_curve
imp = KNNImputer(n_neighbors=1)
x =imp.fit_transform(data)
data = pd.DataFrame(x, columns=all_labels)
#print(pd.isna(data).sum())
X = data[all_labels[:-1]] #data2.iloc[:,:-3] # get the features
y= data[all_labels[len(all_labels)-1]]#data2.iloc[:,-1]#data[all_labels[len(all_labels)-1]] # get the target class
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Evaluate model capability by measuring AUC
roc = list()
clf = GaussianNB()
# Final evaluation with threshold optimization
aucs = list()
bests = list()
for i in range(0,10):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
       train_size=0.7, stratify=y)
   clf.fit(X_train,y_train)
   probs = clf.predict_proba(X_test)[:,1]
   aucs.append(roc_auc_score(y_test, probs))
   fpr, tpr, thr= roc_curve(y_test, probs)
   metrics = [(tp,1-fp, th) for tp, fp, th in zip(tpr, fpr, thr)]
   bests.append(max(metrics, key=lambda tuple: tuple[0]+tuple[1]))
   #print(metrics)
print('best AUC: ',np.max(aucs), 'mean AUC: ', np.mean(aucs),'min AUC: ', np.min(aucs))
best = max(bests,key=lambda tuple:tuple[0]+tuple[1])
best_array=np.array(bests)
print('Max sens: ',np.max(best_array[:,0]),'Min sens: ', np.min(best_array[:,0]))
print('Max spec: ',np.max(best_array[:,1]),'Min spec: ', np.min(best_array[:,1]))
print('J: ', best[0]+best[1]-1,'Threshold: ', best[2],'sensitivity: ', best[0],'specificity: ',best[1])

# %%
