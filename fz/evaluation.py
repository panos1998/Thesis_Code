"""import sklearn modules for train/test spliting and scoring"""
import numpy as np
from sklearn.metrics import  roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def function_evaluation(clf,  X, y,n_iterations: int=10): 
    """This function evaluates a model
Input: a classifier, features array X, label vector y,
 number of iterations n_iterations"""
    aucs = list()
    bests = list()
    for _ in range(0, n_iterations): # for n_iterations times evaluate model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
        train_size=0.7, stratify=y) # train/test 70.30 with stratification
        clf.fit(X_train,y_train) 
        probs = clf.predict_proba(X_test)[:,1] # output the probabilities for postive class
        aucs.append(roc_auc_score(y_test,probs)) # store AUC for every iteration
        fpr,tpr,thr= roc_curve(y_test,probs) # calculate roc curve statistics
        metrics = [(tp,1-fp, th) for tp,fp,th in zip(tpr,fpr,thr)] # save sensitivity, specificity, thr
        bests.append(max(metrics,key=lambda tuple:tuple[0]+tuple[1])) # based on J save sens, spec, thr
   #print(metrics)
    print('best AUC: ',np.max(aucs), 'mean AUC: ', np.mean(aucs),'min AUC: ', np.min(aucs))
    best = max(bests,key=lambda tuple:tuple[0]+tuple[1])# optimal metrics based on best J from all repetitions
    best_array=np.array(bests)
    print('Max sens: ',np.max(best_array[:,0]),'Min sens: ', np.min(best_array[:,0]))# lower and higher sens score
    print('Max spec: ',np.max(best_array[:,1]),'Min spec: ', np.min(best_array[:,1]))# lower and higher spec score
    print('J: ', best[0]+best[1]-1,'Threshold: ', best[2],'sensitivity: ',
    best[0],'specificity: ',best[1])
    return clf # return classifier model
    