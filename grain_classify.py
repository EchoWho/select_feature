import numpy as np
import os,sys
import opt
from bisect import bisect_right
import grain_common

if __name__ == "__main__":
  
  methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]
  #set_id = grain_common.default_set_id
  set_id = int(sys.argv[1])
  target_set_id = set_id
  if len(sys.argv) > 2:
    target_set_id = int(sys.argv[2])
  # Well 'train' here means nothing since set 1 trains on 2,3,4,5 and test on 1.... 
  filename = grain_common.filename_data(target_set_id, 'train')

  print "Load data"
  X_raw, Y = grain_common.load_raw_data(filename)
  X, Y = grain_common.preprocess_X(X_raw, Y, set_id)
  
  model_group_name = grain_common.filename_model(set_id)
  print model_group_name
  d = np.load(model_group_name)
  l = []
  err = []
  costs = []
  if grain_common.ignore_costs:
    groups, group_costs = grain_common.load_group() 
  for method_idx, method in enumerate(methods) :
    l.append([])
    err.append([])
    d_omp = d[method]
    d_selected = d_omp['selected']
    d_w = d_omp['w']
    d_cost = d_omp['cost']
    if grain_common.ignore_costs:
      costs.append([])
      for _, selected in enumerate(d_selected[1:]):
        selected_groups = list(set(groups[selected]))
        costs[-1].append(np.sum(group_costs[selected_groups]))
    else:
      costs.append(d_cost[1:])
    
    for idx, _ in enumerate(d_cost):
      if idx == 0:
        continue
      selected = d_selected[idx] 
      w = d_w[idx]
      selected_X = X[:, selected]
      if selected_X.shape[1] > 0:
        l[-1].append(opt.loss(w, selected_X, Y))
        err[-1].append(np.float64(np.sum(
          (np.sign(Y) * (Y - selected_X.dot(w)) >  0.5))) / 
           Y.shape[0])
      else:
        print "sorry %s" % (method)
        l[-1].append(2)
      
  result_name = grain_common.filename_budget_vs_loss(set_id)
  L = np.array(l)
  costs = np.array(costs)
  np.savez(result_name, 
           L=dict(zip(methods, L)),
           err=dict(zip(methods, np.array(err))),
           costs=dict(zip(methods, costs)), 
           filesize=X.shape[0])
  d.close()
