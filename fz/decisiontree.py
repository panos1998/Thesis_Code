"""decision tree evaluator"""
import subprocess
import numpy as np
from sklearn.tree  import DecisionTreeClassifier
from sklearn import tree
from evaluation import function_evaluation
from parameter_selection import function_parameter_selection
from function_fill_data import function_fill_data
all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm',
'mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge','hearte',
'psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m',
'jphysa','estwt','wstva','chol','hdl','ldl','trig','sys1','dias3',
'fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']
####Find best max depth##################
X, y = function_fill_data()
clf = DecisionTreeClassifier()
params ={}
optimize = 'max_depth'
grid = np.linspace(1, 10, 10)
title = 'AUC per max depth parameter'
function_parameter_selection(clsf=clf, X=X, y=y,params=params,optimize=optimize,
grid=grid,title=title,epochs=100)
#%%
###### Find best min samples_split#################
params ={'max_depth': 4}
optimize = 'min_samples_split'
grid = np.linspace(0.01, 0.1, 10)
title = 'AUC per min samples split parameter'
function_parameter_selection(clsf=clf, X=X, y=y,params=params,optimize=optimize,
grid=grid,title=title,epochs=100)
#%%
##############Find best min samples leaf#############
params ={'max_depth': 4, 'min_samples_split': 0.03}
optimize = 'min_samples_leaf'
grid = np.linspace(0.01, 0.1, 10)
title = 'AUC per min samples leaf parameter'
function_parameter_selection(clsf=clf, X=X, y=y,params=params,optimize=optimize,
grid=grid,title=title,epochs=100)
#final evaluation
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=0.03, min_samples_leaf=0.05)
classifier = function_evaluation(clf, X, y) # evaluate classifier
# export tree graph
dot_data = tree.export_graphviz(classifier, out_file='tree2.dot',feature_names=all_labels[:-1],
class_names=['No', 'Yes'], filled=True, rounded=True,  special_characters=True)
subprocess.run(['dot','-Tpng','tree2.dot','-o','tree4.png'], check=True) # save tree as png
