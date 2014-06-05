import numpy as np
import sys
import opt
from bisect import bisect_right
import vision_common

if __name__ == "__main__":

  filename = sys.argv[1]
  #model_bC_name = sys.argv[2]
  methods = ['OMP'] #, 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE']
  l2_lam = 1e-6

  whiten = False

  X_raw, Y_raw = vision_common.load_raw_data(filename)
  X, Y = vision_common.preprocess_X(X_raw, Y_raw, whiten)
  
  d_pretrain = np.load(vision_common.filename_preprocess_info())
  Y_mean = d_pretrain['Y_mean']
  model_name = vision_common.filename_model(l2_lam)
  print model_name
  d = np.load(model_name)
  l = []
  costs = []
  auc = []
  predicted_val = []
  for method_idx, method in enumerate(methods) :
    l.append([])
    costs.append([])
    auc.append(0)
    predicted_val.append([])
    d_omp = d[method]
    d_selected = d_omp['selected']
    d_w = d_omp['w']
    d_costs = d_omp['cost']
    
    for idx, cost in enumerate(d_costs):
      if cost > 1e8:
        break
      selected = d_selected[idx]
      w = d_w[idx]
      selected_X = X[:, selected]
      if selected_X.shape[1] > 0:
        l[-1].append(opt.loss(w, selected_X, Y))
        costs[-1].append(d_costs[idx])
        auc[-1] += (costs[-1][-1] - costs[-1][-2]) * (l[-1][-2] + l[-1][-1]) / 2.0
        predicted_val[-1].append( (selected_X.dot(w) + Y_mean) )
      else:
        l[-1].append(opt.loss(0, np.zeros(Y.shape[0]), Y))
        costs[-1].append(0)
        predicted_val[-1].append( np.ones(Y.shape[0]) * Y_mean )
    
    auc[-1] /= costs[-1][-1] * l[-1][0] 
    
  result_name = vision_common.filename_budget_vs_loss(l2_lam)
  L = np.array(l)
  costs = np.array(costs)
  auc = np.array(auc)
  np.savez(result_name, L=dict(zip(methods, zip(costs, L, auc, predicted_val))))
  #np.savez(result_name, L=dict(zip(methods, zip(costs, L, auc))))
  d.close()
