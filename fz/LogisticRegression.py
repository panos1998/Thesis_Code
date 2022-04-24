"""logistic regression evaluator"""
from sklearn.linear_model import LogisticRegression
from evaluation import function_evaluation
from weighted_data import weighted_data
#print(pd.isna(data).sum())
X, y = weighted_data()
clf = LogisticRegression(C=100, solver='liblinear', tol=1e-7, max_iter=200)
function_evaluation(clf, X, y) #evaluate classifier
