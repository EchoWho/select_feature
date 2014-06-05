import numpy as np
import opt
import os,sys
import vision_common

filename = vision_common.filename_preprocess_info()
d = np.load(filename)
b = d['b']
C = d['C']

l2_lam = 1e-5
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('yahoo_results_small/uniform.npz', **uniform_results)

# c = np.load('yahoo.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('yahoo_results_small/cost.time.npz', **cost_results)

groups, costs = vision_common.load_group()
group_results = opt.all_results_bC(b, C, costs=costs, groups=groups, optimal=False)
filename = vision_common.filename_model(l2_lam)
np.savez(filename, **group_results)
