"""leicester logistic evaluator"""
from sklearn.linear_model import LogisticRegression
from arfftocsv import processing
from evaluation import function_evaluation
from parameter_selection import function_parameter_selection
all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm',
'mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge','hearte',
'psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m',
'jphysa','estwt','wstva','chol','hdl','ldl','trig','sys1','dias3',
'fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']

labels = ['LeicAge','LeicGender','bmicat','LeicRace','hemda','wstva','LeicHBP','rYdiabe']

to_replace = {'LeicAge': ['50-59', '60-69', '>=70'], 'LeicGender': ['Female', 'Male'],
'bmicat': ["'1.underweight less than 18.5'",
 "'2.normal weight from 18.5 to 24.9'", "'3.pre-obesity from 25 to 29.9'",
 "'4.obesity class 1 from 30 to 34.9'", "'5.obesity class 2 from 35 to 39.9'",
  "'6.obesity class 3 greater than 40'"],
'LeicRace': [0, 6], 'hemda': ["'Not applicable'", 'Yes', 'No'],
 'wstva':[],'LeicHBP': ['No', 'Yes'], 'rYdiabe': ['0.no', '1.yes']
}

values = {'LeicAge': [0, 1, 2], 'LeicGender': [0, 1],
 'bmicat': [1, 2, 3, 4, 5, 6],'LeicRace': [0, 6],
 'hemda': [0, 1, 2], 'wstva':[],'LeicHBP':[0,1],'rYdiabe': [0, 1]}
#prepare the data
path = 'LeicLog.csv'
data = processing(labels=labels, to_replace=to_replace,
all_labels=all_labels, values= values, path=path)
# Apply machine learning techniques
X = data[labels[:-1]] # get the features
y = data[labels[len(labels)-1]] # get the target class
clf = LogisticRegression()
params = {'solver':'liblinear', 'penalty':'l2'}
optimize ='C'
grid = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
title ='AUC per C parameter '
function_parameter_selection(clsf=clf, X=X, y=y,params=params,optimize=optimize,grid=grid,title=title,epochs=100)
clf = LogisticRegression(C=100, solver='liblinear', penalty='l2')
function_evaluation(clf, X, y) # evaluate classifier
