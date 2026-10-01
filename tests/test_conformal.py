import numpy as np
from src.conformal import conformal_pvalues
from src.cost_policy import select_alpha

def test_pvalue_range():
    p=conformal_pvalues([.1,.2,.3],[.05,.25,.5]); assert np.all(p>=0) and np.all(p<=1)

def test_cost_policy_selects_candidate():
    a,t=select_alpha([0,0,1,1],[.5,.2,.01,.08],[.05,.1],10,1); assert a in {.05,.1}; assert len(t)==2
