import numpy as np
import opt
import os
import grain_common

set_id = grain_common.default_set_id
filename = grain_common.filename_preprocess_info(set_id)
d = np.load(filename)
b = d['b']
C = d['C']

l2_lam = 1e-6
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('grain_results_small/uniform.npz', **uniform_results)

# c = np.load('grain.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('grain_results_small/cost.time.npz', **cost_results)

groups, costs = grain_common.load_group()
group_results = opt.all_results_bC(b, C, costs=costs, groups=groups, optimal=False)
filename = grain_common.filename_model(set_id)
np.savez(filename, **group_results)
