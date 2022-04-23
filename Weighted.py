"""weighted ensemble evaluator"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import  RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from arfftocsv import processing
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from evaluation import function_evaluation

all_labels = ['LeicGender','LeicRace','raeducl','mstat','shlt','hlthlm',
'mobilb','lgmusa','grossa','finea','LeicHBP','LeicAge','hearte',
'psyche','bmicat','physActive','drinkd_e','smoken','itot','cfoodo1m',
'jphysa','estwt','wstva','chol','hdl','ldl','trig','sys1','dias3',
'fglu','hba1c','hemda','eatVegFru','everHighGlu','rYdiabe']

to_replace = {'LeicAge': ['50-59', '60-69', '>=70'], 'LeicGender': ['Female', 'Male'],
'bmicat': ["'1.underweight less than 18.5'",
 "'2.normal weight from 18.5 to 24.9'", "'3.pre-obesity from 25 to 29.9'",
 "'4.obesity class 1 from 30 to 34.9'", "'5.obesity class 2 from 35 to 39.9'",
  "'6.obesity class 3 greater than 40'"],
'LeicRace': [0, 6], 'hemda': ["'Not applicable'", 'Yes', 'No'],
 'wstva':[],'LeicHBP': ['No', 'Yes'], 'rYdiabe': ['0.no', '1.yes'],
 'raeducl':["'2.upper secondary and vocational training'",'3.tertiary',
 "'1.less than secondary'",'.o:other',"'.h:missing HSE value'",".m:Missing"],
  'mstat':['3.partnered','1.married','5.divorced','7.widowed','4.separated',
  "'8.never married'"],'shlt':["'2.Very good'",'3.Good','1.Excellent','4.Fair','5.Poor'],
   'hlthlm':['0.no','1.yes','.d:DK'],'hearte':['0.no','1.yes'],
   'psyche':['0.no','1.yes'], 'physActive':['Yes','No'], 'smoken':['0.No',
  '1.Yes',".m:Missing"],'jphysa': ["'1.Sedentary occupation'",
  "'3.Physical work'","'2.Standing occupation'","'.w:not working'",
   "'4.Heavy manual work'",'.m:Missing'], 'everHighGlu':['No',"'Not applicable'",'Yes'],
   'eatVegFru':['Yes', 'No'],'mobilb':[],'lgmusa':[], 'grossa':[],
   'finea':[], 'drinkd_e':[],'itot':[],'cfoodo1m':[], 'estwt':[], 'chol':[],
   'ldl':[], 'hdl':[], 'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
}

values = {'LeicAge': [0, 1, 2], 'LeicGender': [0, 1],
 'bmicat': [1, 2, 3, 4, 5, 6],'LeicRace': [0, 1],
 'hemda': [0, 1, 2], 'wstva':[],'LeicHBP':[0,1],'rYdiabe': [0, 1],
  'raeducl':[2,3,1,0,4,np.nan], 'mstat':[3,1,5,7,4,8],'shlt':[2,3,1,4,5],
   'hlthlm':[0,1,2],'hearte':[0,1], 'psyche':[0,1], 'physActive':[1,0],
  'smoken':[0,1,np.nan], 'jphysa':[1,3,2,5,4,np.nan], 'everHighGlu':[0,2,1],
   'eatVegFru':[1,0],'mobilb':[],'lgmusa':[],'grossa':[], 'finea':[],
    'drinkd_e':[],'itot':[],'cfoodo1m':[], 'estwt':[],'chol':[],'ldl':[], 'hdl':[],
  'trig':[], 'sys1':[], 'dias3':[], 'fglu':[], 'hba1c':[]
 }

data = processing(labels=all_labels, to_replace=to_replace,all_labels=all_labels,
 values= values)
#Filling missing values by mean/mode
data['drinkd_e'] = data['drinkd_e'].fillna(data['drinkd_e'].mean())
data['itot'] = data['itot'].fillna(data['itot'].mean())
data['cfoodo1m'] = data['cfoodo1m'].fillna(data['cfoodo1m'].mean())
data['chol'] = data['chol'].fillna(data['chol'].mean())
data['hdl'] = data['hdl'].fillna(data['hdl'].mean())
data['ldl'] = data['ldl'].fillna(data['ldl'].mean())
data['trig'] = data['trig'].fillna(data['trig'].mean())
data['sys1'] = data['sys1'].fillna(data['sys1'].mean())
data['dias3'] = data['dias3'].fillna(data['dias3'].mean())
data['fglu'] = data['fglu'].fillna(data['fglu'].mean())
data['hba1c'] = data['hba1c'].fillna(data['hba1c'].mean())
data['smoken']=data['smoken'].fillna(data['smoken'].mode()[0])
data['raeducl']=data['raeducl'].fillna(data['raeducl'].mode()[0])
data['jphysa']=data['jphysa'].fillna(data['jphysa'].mode()[0])
####Find best max number of trees ##################
X = data[all_labels[:-1]]# get the features
y= data[all_labels[len(all_labels)-1]] # get the target class

#using auc and sensitivity we must find the pareto front with nsga2
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
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    y_pred = clf.predict_proba(X_test)[:,1]
    auc =roc_auc_score(y_test, y_pred)
    return -auc, -sensitivity
#here genetic algorithm
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=15,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.8, eta=10),
    mutation=get_mutation("real_pm", eta=10),
    eliminate_duplicates=True
)
termination = get_termination('n_gen', 50)
class MyProblem(ElementwiseProblem):
   def __init__(self):
      super().__init__ (n_var=2,
      n_obj=2,
       n_constr=2,
       xl = np.array([0,0]),
       xu= np.array([1,1]))
   
   def _evaluate(self, x, out):
      f1, f2 = objective_function(X, y, weights = x)
      g1 = x[0] + x[1] - 1
      g2 = -x[0] - x[1] + 0.99
      out["F"] = [f1,f2]
      out["G"] = g1,g2

problem = MyProblem()
res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)
Xs = res.X
F= res.F
LR = LogisticRegression(solver='liblinear', max_iter=200, tol=1e-7)
RF = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=0.03,
 min_samples_leaf=0.05)
estimators = [('lr', LR), ('rf', RF)]
weights = Xs[3]
clf= VotingClassifier(estimators=estimators, voting='soft', weights=weights)
plt.scatter(Xs[:,0], Xs[:,1], s=30, facecolors='none', edgecolors='r')
plt.title('Design Space')
plt.show()
plt.scatter(-F[:, 0], -F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()
print(Xs)
function_evaluation(clf, X, y)

