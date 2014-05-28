import numpy as np
import opt
import os
import grain_common
import sys

set_id = int(sys.argv[1]) #grain_common.default_set_id
filename = grain_common.filename_preprocess_info(set_id)
d = np.load(filename)
b = d['b']
C = d['C']
d.close()

if grain_common.whiten:
  d, V = np.linalg.eigh(C)
  d = np.maximum(d, 0)
  D = np.diag(1 / np.sqrt(d + 1e-18))
  L = np.dot(np.dot(V, D), V.T)
  b = np.dot(L.T, b)
  C = np.dot(np.dot(L.T, C), L)

l2_lam = 1e-6
if len(sys.argv) > 2:
  l2_lam = np.float64(sys.argv[2])
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('grain_results_small/uniform.npz', **uniform_results)

# c = np.load('grain.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('grain_results_small/cost.time.npz', **cost_results)

groups, costs = grain_common.load_group()
whiten = grain_common.whiten
ignore_cost = False
if ignore_cost:
  costs = np.ones(costs.shape[0])
group_results = opt.all_results_bC(b, C, costs=costs, groups=groups, optimal=False)
filename = grain_common.filename_model(set_id, l2_lam, whiten, ignore_cost)
np.savez(filename, **group_results)
