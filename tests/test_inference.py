import numpy as np
from src.scoring import local_topk_score, fit_robust, robust_z

def test_topk_and_robust():
    e=np.arange(100,dtype=float).reshape(10,10); assert local_topk_score(e,.1) > e.mean(); s=fit_robust([1,2,3,4,5]); z=robust_z([3],s); assert abs(z[0])<1e-9
