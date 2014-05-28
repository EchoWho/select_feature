import numpy as np
import os,sys
import opt
from bisect import bisect_right
import grain_common

if __name__ == "__main__":
  
  methods = ['OMP'] #, 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]
  #set_id = grain_common.default_set_id
  target_set_id = int(sys.argv[1])
  set_id = int(sys.argv[2])
  l2_lam = np.float64(sys.argv[3])
  # Well 'train' here means nothing since set 1 trains on 2,3,4,5 and test on 1.... 
  filename = grain_common.filename_data(target_set_id, 'train')

  whiten = grain_common.whiten
  ignore_cost = False

  print "Load data"
  X_raw, Y = grain_common.load_raw_data(filename)
  X, Y = grain_common.preprocess_X(X_raw, Y, set_id)
  
  model_group_name = grain_common.filename_model(set_id, l2_lam, whiten, ignore_cost)
  print model_group_name
  d = np.load(model_group_name)
  l = []
  err = []
  costs = []
  auc = []
  if ignore_cost:
    groups, group_costs = grain_common.load_group() 
  for method_idx, method in enumerate(methods) :
    l.append([])
    err.append([])
    auc.append(0)
    costs.append([])
    d_omp = d[method]
    d_selected = d_omp['selected']
    d_w = d_omp['w']
    d_cost = d_omp['cost']

    for idx, _ in enumerate(d_cost):
      selected = d_selected[idx] 
      w = d_w[idx]
      selected_X = X[:, selected]
      if selected_X.shape[1] > 0:
        l[-1].append(opt.loss(w, selected_X, Y))
        err[-1].append(np.float64(np.sum(
          (np.sign(Y) * (Y - selected_X.dot(w)) >  0.5))) / 
           Y.shape[0])
        if ignore_cost:
          costs[-1].append(np.sum(group_costs[np.unique(groups[selected])]))
        else:
          costs[-1].append(d_cost[idx])
        auc[-1] += (costs[-1][-1] - costs[-1][-2]) * (l[-1][-2]  + l[-1][-1]) / 2.0
      else:
        l[-1].append(opt.loss(0, np.zeros(Y.shape[0]), Y))
        err[-1].append(1.0)
        costs[-1].append(0)

  result_name = grain_common.filename_budget_vs_loss(set_id, l2_lam, whiten, ignore_cost)
  L = np.array(l)
  costs = np.array(costs)
  auc = np.array(auc)
  err = np.array(err)
  np.savez(result_name, 
           L=dict(zip(methods, zip(costs, L, auc, err))),
           filesize=X.shape[0])
  d.close()
