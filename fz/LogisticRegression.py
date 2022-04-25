"""logistic regression evaluator"""
from sklearn.linear_model import LogisticRegression
from evaluation import function_evaluation
from function_fill_data import function_fill_data
#print(pd.isna(data).sum())
X, y = function_fill_data()
clf = LogisticRegression(C=100, solver='liblinear', tol=1e-7, max_iter=200)
function_evaluation(clf, X, y) #evaluate classifier
