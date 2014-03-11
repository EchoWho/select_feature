import numpy as np
import sys
import opt
from bisect import bisect_right
import yahoo_common

if __name__ == "__main__":

  filename = sys.argv[1]
  #model_bC_name = sys.argv[2]
  budget_list = [ 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200, 250, 300, \
    350, 400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000,  \
    2500, 3000 ]
  methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]
  set_id = 2

  print "Load data"
  X_raw, Y = yahoo_common.load_raw_data(filename)
  X, Y = yahoo_common.preprocess_X(X_raw, Y, set_id)
  
  for i in range(yahoo_common.n_group_splits):
    print "load group split %d" % (i)
    model_group_name = yahoo_common.filename_model(set_id, i)
    print model_group_name
    d = np.load(model_group_name)
    l = []
    for method_idx, method in enumerate(methods) :
      l.append([])
      d_omp = d[method]
      d_selected = d_omp['selected']
      d_w = d_omp['w']
      d_cost = d_omp['cost']
      
      print "Compute loss %d" % (i)
      for _, budget in enumerate(budget_list) :
        n_groups = bisect_right(d_cost, budget)
        selected = d_selected[n_groups - 1] 
        w = d_w[n_groups - 1]
        selected_X = X[:, selected]
        if selected_X.shape[1] > 0:
          l[-1].append(opt.loss(w, selected_X, Y))
        else:
          print "sorry"
          l[-1].append(2)
      
    result_name = yahoo_common.filename_budget_vs_loss(set_id, i)
    L = np.array(l)
    np.savez(result_name, budget=budget_list, L=dict(zip(methods, L)))
    d.close()
