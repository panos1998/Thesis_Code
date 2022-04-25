"""random forest evaluator"""
from sklearn.ensemble import RandomForestClassifier
from evaluation import function_evaluation
from parameter_selection import function_parameter_selection
from function_fill_data import function_fill_data
####Find best max number of trees ##################
X, y = function_fill_data(categorical=['smoken', 'raeducl', 'jphysa'],
 continuous=['drinkd_e', 'itot', 'cfoodo1m', 'chol',
'hdl', 'ldl', 'trig', 'sys1', 'dias3', 'fglu', 'hba1c'])
#Evaluate model capability by measuring AUC
#for j in range (500):         # TRUE POSITIVE RATE = SENSITIVITY
clf = RandomForestClassifier()
params = {'max_depth':4, 'min_samples_split': 0.03, 'min_samples_leaf': 0.05}
optimize = 'n_estimators'
grid = [100, 200, 500, 1000]
title = 'RF AUC with respect to number of trees'
function_parameter_selection(clsf=clf, X=X, y=y,params=params,optimize=optimize,
grid=grid,title=title)
clf = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_split=0.03,
min_samples_leaf=0.05 )
function_evaluation(clf, X, y) # evaluate classifier
