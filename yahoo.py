import numpy as np
import opt
import os
import yahoo_common

if yahoo_common.whiten :
  print 'data/yahoo.set2.train.bC.npz'
  d = np.load('data/yahoo.set2.train.bC.npz')
else:
  d = np.load('data/yahoo.set2.train.bC.no_whiten.npz')
b = d['b']
C = d['C']

l2_lam = 1e-4
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
  if yahoo_common.whiten:
    print 'yahoo_results/group.%d.npz' % i
    np.savez('yahoo_results/group.%d.npz' % i, **group_results)
  else:
    np.savez('yahoo_results/group.%d.no_whiten.npz' %i, **group_results)
