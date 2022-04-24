import numpy as np
from objective_function import objective_function
from pymoo.core.problem import ElementwiseProblem
from weighted_data import weighted_data
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__ (n_var=2,
        n_obj=2, # auc and sensitivity
        n_constr=2, # w1+w2 =1
        xl = np.array([0,0]), #lower bound
        xu= np.array([1,1])) #upper bound
    def _evaluate(self, x, out,*args, **kwargs):
        f1, f2 = objective_function(weighted_data()[0], weighted_data()[1], weights = x)
        g1 = x[0] + x[1] - 1 # the 2 constraints  for w1, w2
        g2 = -x[0] - x[1] + 0.99
        out["F"] = [f1,f2] # export objectives
        out["G"] = g1,g2 # export constraints