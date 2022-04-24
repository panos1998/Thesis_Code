from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
def objective_function( X, y, weights: list) -> list:
    """This is the vector of objective function to maximize"""
    LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7) 
    RF = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=0.03,
     min_samples_leaf=0.05)
    estimators = [('lr', LR), ('rf', RF)]
    clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    train_size=0.7, random_state=1, stratify=y)
    clf.fit(X_train,y_train)
    y_pred =clf.predict(X_test)
    _, _, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    y_pred = clf.predict_proba(X_test)[:,1]
    auc =roc_auc_score(y_test, y_pred)
    return -auc, -sensitivity