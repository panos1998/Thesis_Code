"""weighted ensemble evaluator"""
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from evaluation import function_evaluation
from MyProblem import MyProblem
from function_fill_data import function_fill_data
#the nsga2 algorithm initialization to solve our optimization problem
################################
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=15,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.8, eta=10), #crossover prob and n
    mutation=get_mutation("real_pm", eta=10), # mutation index
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
LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7)
RF = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=0.03,
 min_samples_leaf=0.05)
estimators = [('lr', LR), ('rf', RF)]
weights = Xs[3] # choose  the fourth non dominated point
clf= VotingClassifier(estimators=estimators, voting='soft', weights=weights)
plt.scatter(Xs[:,0], Xs[:,1], s=30, facecolors='none',
edgecolors='r') # plot solutions
plt.title('Design Space')
plt.show()
plt.scatter(-F[:, 0], -F[:, 1], s=30, facecolors='none',
edgecolors='blue') # plot Pareto optimal front
plt.title("Objective Space")
plt.show()
X, y = function_fill_data()
function_evaluation(clf, X, y) #evaluate weighted classifier
