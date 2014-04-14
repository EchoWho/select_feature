import numpy as np
import sys
import opt
from bisect import bisect_right
import grain_common

if __name__ == "__main__":

  filename = sys.argv[1]
  #model_bC_name = sys.argv[2]
  methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]
  set_id = grain_common.default_set_id

  print "Load data"
  X_raw, Y = grain_common.load_raw_data(filename)
  X, Y = grain_common.preprocess_X(X_raw, Y, set_id)
  
  model_group_name = grain_common.filename_model(set_id)
  print model_group_name
  d = np.load(model_group_name)
  l = []
  costs = []
  for method_idx, method in enumerate(methods) :
    l.append([])
    d_omp = d[method]
    d_selected = d_omp['selected']
    d_w = d_omp['w']
    d_cost = d_omp['cost']
    costs.append(d_cost[1:])
    
    for idx, _ in enumerate(d_cost):
      if idx == 0:
        continue
      selected = d_selected[idx] 
      w = d_w[idx]
      selected_X = X[:, selected]
      if selected_X.shape[1] > 0:
        l[-1].append(opt.loss(w, selected_X, Y))
      else:
        print "sorry %s" % (method)
        l[-1].append(2)
      
  result_name = grain_common.filename_budget_vs_loss(set_id)
  L = np.array(l)
  costs = np.array(costs)
  np.savez(result_name, 
           L=dict(zip(methods, L)),
           costs=dict(zip(methods, costs)), 
           filesize=X.shape[0])
  d.close()
