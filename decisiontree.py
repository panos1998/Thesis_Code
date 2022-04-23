import numpy as np
import matplotlib.pyplot as plt
from asyncio import subprocess
from sklearn.tree  import DecisionTreeClassifier 
from sklearn import tree
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from arfftocsv import processing
import tqdm
import graphviz 
from evaluation import function_evaluation
all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm',
'mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge','hearte',
'psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m',
'jphysa','estwt','wstva','chol','hdl','ldl','trig','sys1','dias3',
'fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']
to_replace = {'LeicAge': ['50-59', '60-69', '>=70'], 'LeicGender': ['Female', 'Male'], 
'bmicat': ["'1.underweight less than 18.5'",
 "'2.normal weight from 18.5 to 24.9'", "'3.pre-obesity from 25 to 29.9'",
 "'4.obesity class 1 from 30 to 34.9'", "'5.obesity class 2 from 35 to 39.9'",
  "'6.obesity class 3 greater than 40'"],
'LeicRace': [0, 6], 'hemda': ["'Not applicable'", 'Yes', 'No'],
 'wstva':[],'LeicHBP': ['No', 'Yes'], 'rYdiabe': ['0.no', '1.yes'],
 'raeducl':["'2.upper secondary and vocational training'",'3.tertiary',
 "'1.less than secondary'",'.o:other',"'.h:missing HSE value'",".m:Missing"],
  'mstat':['3.partnered','1.married','5.divorced','7.widowed','4.separated',
  "'8.never married'"],'shlt':["'2.Very good'",'3.Good','1.Excellent','4.Fair','5.Poor'],
   'hlthlm':['0.no','1.yes','.d:DK'],'hearte':['0.no','1.yes'],
   'psyche':['0.no','1.yes'], 'physActive':['Yes','No'], 'smoken':['0.No',
  '1.Yes',".m:Missing"],'jphysa': ["'1.Sedentary occupation'",
  "'3.Physical work'","'2.Standing occupation'","'.w:not working'",
   "'4.Heavy manual work'",'.m:Missing'], 'everHighGlu':['No',"'Not applicable'",'Yes'],
   'eatVegFru':['Yes', 'No'],'mobilb':[],'lgmusa':[], 'grossa':[],
   'finea':[], 'drinkd_e':[],'itot':[],'cfoodo1m':[], 'estwt':[], 'chol':[],
   'ldl':[], 'hdl':[], 'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
}

values = {'LeicAge': [0, 1, 2], 'LeicGender': [0, 1],
 'bmicat': [1, 2, 3, 4, 5, 6],'LeicRace': [0, 1],
 'hemda': [0, 1, 2], 'wstva':[],'LeicHBP':[0,1],'rYdiabe': [0, 1],
  'raeducl':[2,3,1,0,4,np.nan], 'mstat':[3,1,5,7,4,8],'shlt':[2,3,1,4,5],
   'hlthlm':[0,1,2],'hearte':[0,1], 'psyche':[0,1], 'physActive':[1,0],
  'smoken':[0,1,np.nan], 'jphysa':[1,3,2,5,4,np.nan], 'everHighGlu':[0,2,1],
   'eatVegFru':[1,0],'mobilb':[],'lgmusa':[],'grossa':[], 'finea':[],
    'drinkd_e':[],'itot':[],'cfoodo1m':[], 'estwt':[],'chol':[],'ldl':[], 'hdl':[],
  'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
 }
data = processing(labels=all_labels, to_replace=to_replace,all_labels=all_labels,
 values= values)

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

####Find best max depth##################
X = data[all_labels[:-1]] 
y= data[all_labels[len(all_labels)-1]]
#Evaluate model capability by measuring AUC
aucs = np.zeros((10, 10, 100)) # initialize an array to store aucs
#for j in range (500):         # TRUE POSITIVE RATE = SENSITIVITY
for k in tqdm.tqdm(range(100), colour='CYAN'): # for 100 epochs
    for i in range(0,10):
        #run through 10 different stratified datasets
        j = 0# TRUE NEGATIVE RATE = SPECIFICITY
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,
        test_size=0.3,stratify=y)# stratified train/test split 70/30
        for d in np.linspace(1, 10, 10):
            #evaluate each dataset over 5 different C values
            clf = DecisionTreeClassifier(max_depth=d).fit(X_train, y_train)#for each C train
            y_pred = (clf.predict_proba(X_test))[:,1]# get prob predictions
            fpr, sensitivity, thresholds = roc_curve(y_test, y_pred) # get metrics
            aucs[i, j, k] =(roc_auc_score(y_test, y_pred))#3-order tensor,auc for each C
            j = j + 1# for each dataset for each epoch
means = np.mean(np.mean(aucs, axis =2), axis =0)#mean auc per c over all datasets and epochs
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
    for i in range(0,10): #run through 10 different stratified datasets
       j = 0 #TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y,
       train_size= 0.7,test_size=0.3, stratify=y)#stratified train/test split 70/30
       for mss in np.linspace(0.01,0.1,10):#evaluate each dataset over 5 different C values
           clf = DecisionTreeClassifier(max_depth=4, min_samples_split=mss).fit(X_train,
            y_train) # train classifier over a dataset for C values
           y_pred = (clf.predict_proba(X_test))[:,1]#get prob predictions
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred) # get metrics
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred))#3-order tensor,auc for each C
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
       j = 0 #TRUE NEGATIVE RATE = SPECIFICITY
       X_train, X_test, y_train, y_test = train_test_split(X, y,
       train_size= 0.7,test_size=0.3, stratify=y) # stratified train/test split 70/30
       for msl in np.linspace(0.01,0.1,10):#evaluate each dataset over 5 different C values
           clf = DecisionTreeClassifier(max_depth=4, min_samples_split=0.03,
            min_samples_leaf=msl).fit(X_train, y_train)#train classifier over a dataset for C values
           y_pred = (clf.predict_proba(X_test))[:,1]#get prob predictions
           fpr, sensitivity, thresholds = roc_curve(y_test, y_pred)#get metrics
           aucs[i, j, k] =(roc_auc_score(y_test, y_pred))#3-order tensor saves auc for each C
           j = j + 1                                      # for each dataset for each epoch
means = np.mean(np.mean(aucs, axis =2), axis =0)# mean auc per c over all datasets and epochs
print(means)
ax[2].set_xlabel('Min samples split')
ax[2].plot(np.linspace(0.01,0.1,10), means)
aucs = list()
bests = list()
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=0.03, min_samples_leaf=0.05)
classifier = function_evaluation(clf, X, y)
dot_data = tree.export_graphviz(classifier, out_file='tree2.dot',feature_names=all_labels[:-1]
,class_names=['No', 'Yes'], filled=True, rounded=True,  special_characters=True)  
subprocess.run(['dot','-Tpng','tree2.dot','-o','tree4.png'],check=True )
graph = graphviz.Source(dot_data) 
fig.tight_layout()
plt.show()
