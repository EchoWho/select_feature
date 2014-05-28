import numpy as np
import opt
import os,sys
import yahoo_common

set_id = int(sys.argv[1])
group_size = int(sys.argv[2])
filename = yahoo_common.filename_preprocess_info(set_id)
d = np.load(filename)
b = d['b']
C = d['C']

l2_lam = 1e-6
if len(sys.argv) > 3:
  l2_lam = np.float64(sys.argv[3])
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('yahoo_results_small/uniform.npz', **uniform_results)

# c = np.load('yahoo.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('yahoo_results_small/cost.time.npz', **cost_results)

groups, costs = yahoo_common.load_group(group_size)
whiten = yahoo_common.whiten
ignore_cost = False
if ignore_cost:
  costs = np.array([ (cost >= 1e7) * cost for _, cost in enumerate(costs)])
  costs = np.maximum(costs, 1)
for i in range(yahoo_common.n_group_splits):
  group_results = opt.all_results_bC(b, C, costs=costs[i], groups=groups[i], optimal=False)
  filename = yahoo_common.filename_model(set_id, i, group_size, l2_lam, whiten, ignore_cost)
  np.savez(filename, **group_results)
