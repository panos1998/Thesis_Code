#%%
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
def function_parameter_selection(clsf, X: List[List], y: List, params: Dict,
 optimize: str, grid: List, title: str, epochs: int=10 )-> List:
    aucs = np.zeros((10, len(grid), epochs)) # initialize an array to store aucs
    for k in tqdm.tqdm(range(epochs), colour='CYAN'):
        for i in range(0,10):
        # run through 10 different stratified datasets
            j = 0
            X_train, X_test, y_train, y_test = train_test_split(X, y,
            train_size= 0.7,test_size=0.3, stratify=y) # stratified train/test split 70/30
            for n in grid:
                #evaluate each dataset over 4 different n_trees values
                params[optimize] = n
                clsf = clsf.set_params(**params).fit(X_train, y_train) #train over a dataset for ntree values
                y_pred = (clsf.predict_proba(X_test))[:,1] # get prob predictions
                aucs[i, j, k] =(roc_auc_score(y_test, y_pred)) #3-order tensor saves auc for each n_trees
                j = j + 1                                      # for each dataset for each epoch
    means = np.mean(np.mean(aucs, axis =2), axis =0)#mean auc per n_trees over all datasets and epochs
    print(means)
    plt.title(title)
    plt.xlabel(optimize)
    plt.ylabel('Mean AUC')
    plt.plot(grid, means)
    plt.show()
# %%
