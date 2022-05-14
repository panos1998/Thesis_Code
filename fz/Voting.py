#%%
"""voting evaluator"""
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from evaluation import function_evaluation
from function_fill_data import function_fill_data

X, y = function_fill_data(categorical=['smoken', 'raeducl', 'jphysa'],
 continuous=['drinkd_e', 'itot', 'cfoodo1m', 'chol',
'hdl', 'ldl', 'trig', 'sys1', 'dias3', 'fglu', 'hba1c'])
LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7) # first classifier
RF = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=0.03,
 min_samples_leaf=0.05) # second classifier
estimators = [('lr', LR), ('rf', RF)]# classifer pool
clf = VotingClassifier(estimators=estimators, voting='soft')#ensebmle voting  classifer with soft voting
function_evaluation(clf, X, y) # evaluate classifer

# %%
