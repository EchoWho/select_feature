import numpy as np
import sys
import opt
from bisect import bisect_right
import yahoo_common
import os
from opt_util import ndcg_overall

if __name__ == "__main__":

  filename = sys.argv[1]
  #model_bC_name = sys.argv[2]
  methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE']
  set_id = int(sys.argv[2])
  group_size = int(sys.argv[3])
  l2_lam = np.float64(sys.argv[4])

  filename_querystarts = "{}.querystarts.npz".format(os.path.splitext(filename)[0])
  query_starts = np.load(filename_querystarts)['query_starts']
  print len(query_starts)

  whiten = yahoo_common.whiten
  ignore_cost = False # This is true if we set all groups with equal cost.

  X_raw, Y = yahoo_common.load_raw_data(filename)
  print "loaded"
  print np.max(Y), np.min(Y)
  X, Y_mean = yahoo_common.preprocess_X_rebut(X_raw, Y, set_id, whiten)
  print Y_mean

  print np.sum(np.array([np.max(Y[query_starts[i]:query_starts[i+1]]) for i in range(len(query_starts)-1)]) == 0)

  print "preprocessed"
  
  if ignore_cost:
    vec_groups, vec_group_costs = yahoo_common.load_group(group_size)
  for i in range(yahoo_common.n_group_splits):
    if ignore_cost:
      groups = vec_groups[i]
      group_costs = vec_group_costs[i]
    model_group_name = yahoo_common.filename_model(set_id, i, group_size, l2_lam, 
      whiten, ignore_cost)
    print model_group_name
    d = np.load(model_group_name)
    l = []
    costs = []
    #area under the curve
    auc = []
    for method_idx, method in enumerate(methods) :
      l.append([])
      costs.append([])
      auc.append(0)
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
          Y_hat = selected_X.dot(w) + Y_mean
#          l[-1].append(opt.loss(w, selected_X, Y))
## NDCG here
          ndcg_val = ndcg_overall(selected_X.dot(w), Y, query_starts)
          print ndcg_val
          l[-1].append(ndcg_val)

          if ignore_cost:
            costs[-1].append(np.sum(group_costs[np.unique(groups[selected])]))
          else:
            costs[-1].append(d_costs[idx])
          auc[-1] += (costs[-1][-1] - costs[-1][-2]) * (l[-1][-2] + l[-1][-1]) / 2.0
        else:
          l[-1].append(opt.loss(0, np.zeros(Y.shape[0]), Y))
          costs[-1].append(0)
      
      auc[-1] /= costs[-1][-1] * l[-1][0] 
      
    result_name = yahoo_common.filename_budget_vs_loss(set_id, i, group_size, l2_lam,
      whiten, ignore_cost)
    result_name = "{}_rebut.npz".format(os.path.splitext(result_name)[0])
    L = np.array(l)
    costs = np.array(costs)
    auc = np.array(auc)
    np.savez(result_name, L=dict(zip(methods, zip(costs, L, auc))))
    d.close()
