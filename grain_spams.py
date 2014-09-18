import spams
import numpy as np
import scipy as sci
import yahoo_common
import grain_common
import sys,os
import opt


def spams_train(X, Y, groups, costs, lambda1 = 0.1):
  spams_tree, spams_params = yahoo_common.create_spams_params(groups, costs)
  W0 = np.zeros((X.shape[1], Y.shape[1]),dtype=np.float64,order="FORTRAN")
  spams_params['lambda1'] = lambda1
  return spams.fistaTree(Y, X, W0, spams_tree, True, **spams_params)
  
def load_data(set_id):
  groups, costs = grain_common.load_group()

  train_set_id = set_id % 5 + 1
  filename = grain_common.filename_data(train_set_id, 'train')
  X_raw, Y_raw = grain_common.load_raw_data(filename)
  X, Y = grain_common.preprocess_X(X_raw, Y_raw, set_id)
  X,Y = yahoo_common.convert_to_spams_format(X, Y, groups)

  filename = grain_common.filename_data(set_id, 'train')
  X_raw, Y_raw = grain_common.load_raw_data(filename)
  X_tes, Y_tes = grain_common.preprocess_X(X_raw, Y_raw, set_id)
  X_tes,Y_tes = yahoo_common.convert_to_spams_format(X_tes, Y_tes, groups)
  print "finished loading..."
  return X, Y, X_tes, Y_tes

def batch_all_old(X,Y, X_tes, Y_tes, set_id):
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
  np.savez('grain_results/spams_%d_%d.npz' % (set_id), v_W=v_W, v_optim_info=v_optim_info, budgets=budget, losses=loss)

def batch_all(X,Y,X_tes,Y_tes, set_id, v_lams):
  v_lams = np.array(sorted(v_lams)[::-1])
  loss = np.zeros(len(v_lams))
  budget = np.zeros(len(v_lams))
  err = np.zeros(len(v_lams))
  models = []
  for i, lam in enumerate(v_lams):
    loss[i], budget[i], err[i], model= train_test_one(X,Y,X_tes,Y_tes,set_id, lam)
    models.append(model)
  models = np.array(models)

  np.savez('grain_results/spams_%d.npz' % (set_id), budget=budget, loss=loss, err=err, models=models)


def train_test_one(X,Y,X_tes,Y_tes,set_id, lam):
  groups, costs = grain_common.load_group()
  sorted_groups = np.array(sorted(groups))
  (W, optim_info) = spams_train(X, Y, groups, costs, lam)
  loss = opt.loss(W[:,0], X_tes, Y_tes) 
  budget = np.sum(costs[ list(set(sorted_groups[np.nonzero(W)[0]])) ])
  

  err = np.sum((np.sign(Y_tes) * (Y_tes - X_tes.dot(W[:,0]))) > 0.5) / np.float64(len(Y_tes))  
  np.savez('grain_results/spams_%d_lam_%f.npz' % (set_id, lam), budget=budget, loss=loss, err=err, models=W[:,0])

  return loss, budget, err, W[:,0]


if __name__ == "__main__":
  set_id = int(sys.argv[1])
  X, Y, X_tes, Y_tes = load_data(set_id)
  nbr_train = X.shape[0]

  v_lams = np.array([ 1e3, 500, 1e2, 50, 25, 1e1, 6, 4, 3, 2.5, 2, 1.5, 1, 1e-1, 1e-2 ]) * nbr_train
  batch_all(X, Y, X_tes, Y_tes.ravel(), set_id, v_lams)
 # v_lams = np.array([ 2 ]) * nbr_train
 # loss, budget = train_test_one(X,Y, X_tes,Y_tes, v_lams[0])
 # print loss
 # print budget
