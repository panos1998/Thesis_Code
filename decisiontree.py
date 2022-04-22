#%%
from turtle import width
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
#Filling missing values by mean/mode 
#data.fillna(
  # data[['drinkd_e','itot','cfoodo1m','chol','hdl','ldl','trig','sys1','dias3', 'fglu','hba1c']].mean(), inplace=True)
data['drinkd_e'] = data['drinkd_e'].fillna(data['drinkd_e'].mean())
data['itot'] = data['itot'].fillna(data['itot'].mean())
data['cfoodo1m'] = data['cfoodo1m'].fillna(data['cfoodo1m'].mean())
data['chol'] = data['chol'].fillna(data['chol'].mean())
data['hdl'] = data['hdl'].fillna(data['hdl'].mean())
data['ldl'] = data['ldl'].fillna(data['ldl'].mean())
data['trig'] = data['trig'].fillna(data['trig'].mean())
data['sys1'] = data['sys1'].fillna(data['sys1'].mean())
data['dias3'] = data['dias3'].fillna(data['dias3'].mean())
data['fglu'] = data['fglu'].fillna(data['fglu'].mean())
data['hba1c'] = data['hba1c'].fillna(data['hba1c'].mean())
data['smoken']=data['smoken'].fillna(data['smoken'].mode()[0])
data['raeducl']=data['raeducl'].fillna(data['raeducl'].mode()[0])
data['jphysa']=data['jphysa'].fillna(data['jphysa'].mode()[0])
#data2 = data.dropna(axis =1 )

"""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
imp = KNNImputer(n_neighbors=3)
x =imp.fit_transform(data)
data = pd.DataFrame(x, columns=all_labels)
#print(pd.isna(data).sum())
"""
####Find best max depth##################
X = data[all_labels[:-1]] #data2.iloc[:,:-3] # get the features
y= data[all_labels[len(all_labels)-1]]#data2.iloc[:,-1]#data[all_labels[len(all_labels)-1]] # get the target class
from sklearn.tree  import DecisionTreeClassifier 
from sklearn import tree
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#Evaluate model capability by measuring AUC
import graphviz 
aucs = np.zeros((10, 10, 100)) # initialize an array to store aucs
#for j in range (500):         # TRUE POSITIVE RATE = SENSITIVITY
for k in tqdm.tqdm(range(100), colour='CYAN'): # for 100 epochs
    for i in range(0,10): # run through 10 different stratified datasets
       j = 0                                       # TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.7,test_size=0.3, stratify=y) # stratified train/test split 70/30
       for d in np.linspace(1, 10, 10):#evaluate each dataset over 5 different C values
           clf = DecisionTreeClassifier(max_depth=d).fit(X_train, y_train) # train classifier over a dataset for C values
           y_pred = (clf.predict_proba(X_test))[:,1] # get prob predictions
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred) # get metrics
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred)) #3-order tensor saves auc for each C
           j = j + 1                                      # for each dataset for each epoch
means = np.mean(np.mean(aucs, axis =2), axis =0) # mean auc per c over all datasets and epochs
print(means)
fig, ax = plt.subplots(3,1)
ax[0].set_xlabel('Max depth')
ax[0].set_ylabel('Mean AUC')
ax[0].plot(np.linspace(1,10,10), means)
#%%
###### Find best min samples_split#################
aucs = np.zeros((10, 10, 100))
means = list()
for k in tqdm.tqdm(range(100), colour='CYAN'): # for 100 epochs
    for i in range(0,10): # run through 10 different stratified datasets
       j = 0                                       # TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.7,test_size=0.3, stratify=y) # stratified train/test split 70/30
       for mss in np.linspace(0.01,0.1,10):#evaluate each dataset over 5 different C values
           clf = DecisionTreeClassifier(max_depth=4, min_samples_split=mss).fit(X_train, y_train) # train classifier over a dataset for C values
           y_pred = (clf.predict_proba(X_test))[:,1] # get prob predictions
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred) # get metrics
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred)) #3-order tensor saves auc for each C
           j = j + 1                                      # for each dataset for each epoch
means = np.mean(np.mean(aucs, axis =2), axis =0) # mean auc per c over all datasets and epochs
print(means)
ax[1].set_xlabel('Min samples split')
ax[1].plot(np.linspace(0.01,0.1,10), means)
#%%
##############Find best min samples leaf#############
aucs = np.zeros((10, 10, 100))
means = list()
for k in tqdm.tqdm(range(100), colour='CYAN'): # for 100 epochs
    for i in range(0,10): # run through 10 different stratified datasets
       j = 0                                       # TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.7,test_size=0.3, stratify=y) # stratified train/test split 70/30
       for msl in np.linspace(0.01,0.1,10):#evaluate each dataset over 5 different C values
           clf = DecisionTreeClassifier(max_depth=4, min_samples_split=0.03, min_samples_leaf=msl).fit(X_train, y_train) # train classifier over a dataset for C values
           y_pred = (clf.predict_proba(X_test))[:,1] # get prob predictions
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred) # get metrics
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred)) #3-order tensor saves auc for each C
           j = j + 1                                      # for each dataset for each epoch
means = np.mean(np.mean(aucs, axis =2), axis =0) # mean auc per c over all datasets and epochs
print(means)
ax[2].set_xlabel('Min samples split')
ax[2].plot(np.linspace(0.01,0.1,10), means)
#%%
##########################
roc = list()
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=0.03, min_samples_leaf=0.05)
"""
for i in tqdm.tqdm(range(0,10)):
   X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, train_size=0.7, stratify=y)
   clf.fit(X_train,y_train)
   y_pred =clf.predict_proba(X_test)[:,1]
   roc.append(roc_auc_score(y_test, y_pred))
print(np.mean(roc))
"""
"""
roc = list()
scores = list()
#Find- proof the best threshold closest to paper
thresholds = np.linspace(0, 1, 100)
for thr in tqdm.tqdm(thresholds):
   younden = list()
   for i in range(0,10):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
       train_size=0.7, stratify=y)
      clf.fit(X_train,y_train)
      y_pred =(clf.predict_proba(X_test)[:,1]>=thr).astype(bool)
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
      specificity = tn/(tn+fp)
      sensitivity = tp/(tp+fn)
      younden.append([specificity+sensitivity-1,specificity, sensitivity])
   scores.append((sum(you[0] for you in younden)/len(younden),
   sum(you[1] for you in younden)/ len(younden), # per threshold
   sum(you[2] for you in younden)/ len(younden), thr))
optimal = max(scores, key=lambda score: score[0])
print('Maximum younden,specificity, sensitivity, threshold ', optimal)

# Final evaluation with threshold optimization
younden = list()
for i in range(0, 10):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
       train_size=0.7, stratify=y)
      clf.fit(X_train,y_train)
      y_pred =(clf.predict_proba(X_test)[:,1]>=optimal[3]).astype(bool) #0.008 πολυ καλο
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
      specificity = tn/(tn+fp)
      sensitivity = tp/(tp+fn)
      younden.append([specificity+sensitivity-1,specificity, sensitivity])
younden = np.array(younden)
mean = np.mean(younden, axis=0)
print(mean)
print('Max specificity: ',np.max(younden[:,1]), ' Max sensitivity: ', np.max(younden[:,2]))
print('Min specificity: ',np.min(younden[:,1]), ' Min sensitivity: ', np.min(younden[:,2]))
"""


# %%
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
dot_data = tree.export_graphviz(clf, out_file='tree2.dot', 
                     feature_names=all_labels[:-1],  
                     class_names=['No', 'Yes'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 
fig.tight_layout()
fig