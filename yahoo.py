import numpy as np
import opt
import os
import yahoo_common

set_id = 2
filename = yahoo_common.filename_preprocess_info(set_id)
d = np.load(filename)
b = d['b']
C = d['C']

l2_lam = 1e-6
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('yahoo_results_small/uniform.npz', **uniform_results)

# c = np.load('yahoo.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('yahoo_results_small/cost.time.npz', **cost_results)

groups, costs = yahoo_common.load_group()
for i in range(yahoo_common.n_group_splits):
  group_results = opt.all_results_bC(b, C, costs=costs[i], groups=groups[i], optimal=False)
  filename = yahoo_common.filename_model(set_id, i)
  np.savez(filename, **group_results)
