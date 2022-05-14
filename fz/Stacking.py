#%%
"""stacking algorithm evaluation"""
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from evaluation import function_evaluation
from function_fill_data import function_fill_data

X, y = function_fill_data(categorical=['smoken', 'raeducl', 'jphysa'],
 continuous=['drinkd_e', 'itot', 'cfoodo1m', 'chol',
'hdl', 'ldl', 'trig', 'sys1', 'dias3', 'fglu', 'hba1c'])
LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7)# first base estimator
RF = RandomForestClassifier(n_estimators=200, max_depth=4,
 min_samples_split=0.03,# second base  estimator
min_samples_leaf=0.05)
metaRF = RF
estimators = [('lr', LR), ('rf', RF)] # estimators pool
clf = StackingClassifier(estimators=estimators,
final_estimator=metaRF, cv=5)# ensemble stackig estimator
function_evaluation(clf, X, y) # evaluate classifer
# %%
