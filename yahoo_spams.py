import spams
import numpy as np
import scipy as sci
import yahoo_common
import sys,os
import opt


def spams_train(X, Y, groups, costs, lambda1 = 0.1):
  spams_tree, spams_params = yahoo_common.create_spams_params(groups, costs)
  W0 = np.zeros((X.shape[1], Y.shape[1]),dtype=np.float64,order="FORTRAN")
  spams_params['lambda1'] = lambda1
  return spams.fistaTree(Y, X, W0, spams_tree, True, **spams_params)
  
def load_data(set_id, group_size):
  #set_id = int(sys.argv[1])
  #group_size = int(sys.argv[2])

  v_groups, v_costs = yahoo_common.load_group(group_size)
  groups = v_groups[0]
  costs = v_costs[0]

  filename = yahoo_common.filename_data(set_id, 'train')
  X_raw, Y_raw = yahoo_common.load_raw_data(filename)
  X, Y = yahoo_common.preprocess_X(X_raw, Y_raw, set_id, False)
  X,Y = yahoo_common.convert_to_spams_format(X, Y, groups)

  filename = yahoo_common.filename_data(set_id, 'test')
  X_raw, Y_raw = yahoo_common.load_raw_data(filename)
  X_tes, Y_tes = yahoo_common.preprocess_X(X_raw, Y_raw, set_id, False)
  X_tes,Y_tes = yahoo_common.convert_to_spams_format(X_tes, Y_tes, groups)
  print "finished loading..."
  return X, Y, X_tes, Y_tes

def batch_all_old(X,Y, X_tes, Y_tes, set_id, group_size):
  nbr_train = X.shape[0]
  v_lam = np.array([ 1e-1, 1e-2, 
                     6.7e-3, 3.3e-3, 1e-3, 
                     8e-4, 6e-4, 4e-4, 2e-4, 1e-4, 
                     8e-5, 5e-5, 1e-5 ]) * nbr_train
  budget = np.zeros(len(v_lam))
  loss = np.zeros(len(v_lam))
  sorted_groups = np.array(sorted(groups))
  v_W = []
  v_optim_info = []
  for i, lam in enumerate(v_lam):
    (W, optim_info) = spams_train(X, Y, groups, costs, lam)
    budget[i] = np.sum(costs[ list(set(sorted_groups[np.nonzero(W)[0]])) ])
    loss[i] = opt.loss(W[:,0], X_tes, Y_tes[:,0]) 
    v_W.append(W)
    v_optim_info.append(optim_info)
  np.savez('yahoo_results/spams_%d_%d.npz' % (set_id, group_size), v_W=v_W, v_optim_info=v_optim_info, budgets=budget, losses=loss)

def batch_all(X,Y,X_tes,Y_tes, set_id, group_size, v_lams):
  v_lams = np.array(sorted(v_lams)[::-1])
  loss = np.zeros(len(v_lams))
  budget = np.zeros(len(v_lams))
  for i, lam in enumerate(v_lams):
    loss[i], budget[i] = train_test_one(X,Y,X_tes,Y_tes,set_id, group_size, lam)

  np.savez('yahoo_results/spams_%d_%d.npz' % (set_id, group_size), budget=budget, loss=loss)


def train_test_one(X,Y,X_tes,Y_tes,set_id, group_size, lam):
  v_groups, v_costs = yahoo_common.load_group(group_size)
  groups = v_groups[0]
  sorted_groups = np.array(sorted(groups))
  costs = v_costs[0]
  (W, optim_info) = spams_train(X, Y, groups, costs, lam)
  loss = opt.loss(W[:,0], X_tes, Y_tes[:,0]) 
  budget = np.sum(costs[ list(set(sorted_groups[np.nonzero(W)[0]])) ])
  return loss, budget


if __name__ == "__main__":
  set_id = int(sys.argv[1])
  group_size = int(sys.argv[2])
  X, Y, X_tes, Y_tes = load_data(set_id, group_size)
  nbr_train = X.shape[0]

  v_lams = np.array([ 1e-1, 1e-2,  6.7e-3, 3.3e-3, 1e-3, 8e-4, 6e-4, 4e-4, 2e-4, 1e-4, 8e-5, 5e-5, 1e-5 ]) * nbr_train
  batch_all(X, Y, X_tes, Y_tes, set_id, group_size, v_lams)
