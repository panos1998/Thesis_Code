#%%
"""weighted ensemble evaluator"""
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from evaluation import function_evaluation
from MyProblem import MyProblem
from function_fill_data import function_fill_data
import numpy as np
#the nsga2 algorithm initialization to solve our optimization problem
################################
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=15,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.8, eta=10), #crossover prob and n
    mutation=get_mutation("real_pm",prob = 0.8, eta=10), # mutation index
    eliminate_duplicates=True
)
termination = get_termination('n_gen', 50)
##############################
problem = MyProblem() # initialize problem
############################
res = minimize(problem, algorithm, termination, seed=1,
save_history=True, verbose=True)  # start solving
Xs = res.X #  the non dominated solutions in decision space
F= res.F # the non dominated points in objective space
#%%
LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7)
RF = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=0.03,
 min_samples_leaf=0.05)
estimators = [('lr', LR), ('rf', RF)]
#%%
plt.scatter(Xs[:,0], Xs[:,1], s=30, facecolors='none',
edgecolors='r') # plot solutions
plt.title('Design Space')
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()
plt.scatter(-F[:, 0], -F[:, 1], s=30, facecolors='none',
edgecolors='blue') # plot Pareto optimal front
plt.title("Objective Space")
plt.xlabel("AUC")
plt.ylabel('Sensitivity')
plt.show()
#%%

weights = Xs[4] # choose  the fourth non dominated point
clf= VotingClassifier(estimators=estimators, voting='soft', weights=weights)

X, y = function_fill_data(categorical=['smoken', 'raeducl', 'jphysa'],
 continuous=['drinkd_e', 'itot', 'cfoodo1m', 'chol',
'hdl', 'ldl', 'trig', 'sys1', 'dias3', 'fglu', 'hba1c'])
clf =function_evaluation(clf, X, y) #evaluate weighted classifier
probs = clf.predict_proba(X)
target = [0.3 +np.random.random_sample()/10 if _==0 else 0.7+ np.random.random_sample()/10 for _ in y]
plt.scatter(target,probs[:,1], s=6, facecolors='none',
edgecolors='black')
plt.xlim([0,1])
plt.xticks([0.35,0.75],['Healthy', 'Diseased'])
plt.ylabel('Cut-off point')
plt.yticks([0,0.2,0.4,0.6,0.8])
plt.hlines(y=0.193,xmin=0,xmax=1,linestyles='dashed')
plt.show()
data = [(prob,label) for prob,label in zip(probs[:,1],y)]
class0=[tuple for tuple in data if tuple[1]==0]
class1=[tuple for tuple in data if tuple[1]==1]
sb.kdeplot([np.round(point[0],2).astype(float) for point in class0],fill=True,bw=0.35)
sb.kdeplot([np.round(point[0],2).astype(float) for point in class1],fill=True,bw=0.35)
plt.vlines(x=0.193,ymin=0,ymax=5.5,linestyles='dashed')
plt.xlabel('Cut-off point')
plt.show()
# %%
