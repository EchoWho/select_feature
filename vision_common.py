import numpy as np
import scipy.sparse as ssp
import scipy.io

n_group_splits=1
feat_dim=2400
whiten=False

root_dir='/home/hanzhang/code/select_feature'
data_dir='%s/vision_data' % (root_dir)
result_dir='%s/vision_results' % (root_dir)

def param_str(do_whiten=whiten, ignore_cost=False) :
  s = ''
  if not do_whiten :
    s += 'no_whiten.'
  if ignore_cost :
    s += 'eq_costs.'
  return s

def filename_data(mode) :
  """ Training data filename """
  return '%s/%s.mat' % (data_dir, mode)

def filename_preprocess_info(do_whiten=whiten) : 
  """ 
    Filename containing information needed for preprocess data, 
    and (b, C) for training 
  """
  return '%s/vision.train.bC.%snpz' % (data_dir, param_str(do_whiten))
 
def filename_model(l2_lam=1e-6, do_whiten=whiten, ignore_cost=False) :
  """ Filename for model trained by feature selection """
  return '%s/model.lam%f.%snpz' % (result_dir, l2_lam, param_str(do_whiten, ignore_cost))

def filename_budget_vs_loss(l2_lam=1e-6, do_whiten=whiten, ignore_cost=False) :
  """ Filenmae for final results """
  return '%s/budget_vs_loss.lam%f.%snpz' % (result_dir, l2_lam, param_str(do_whiten, ignore_cost))

def preprocess_X(X_raw, Y_raw, do_whiten=whiten) :
  filename = filename_preprocess_info(do_whiten)
  d = np.load(filename)
  X_mean = d['X_mean']
  X_std = d['X_std']
  X = (X_raw - X_mean) / X_std
  Y = Y_raw - d['Y_mean']
  d.close()
  return X, Y

def load_group() :
  d_groups = scipy.io.loadmat('%s/groups.mat' % (data_dir))
  groups = d_groups['groups'][0, :]
  costs = d_groups['costs'][0, :]
  return groups, costs

def load_raw_data(filename) :
  d_raw = scipy.io.loadmat(filename)
  X = d_raw['X']
  Y = d_raw['Y'][0, :]
  return X, Y

def convert_to_spams_format(X, Y, groups):
  # X, Y are preprocessed
  group_names = sorted(list(set(groups)))
  selected_feats = [np.array([], dtype=np.int)]
  selected_feats += [ np.nonzero(groups == g)[0] for g in group_names ]
  selected_feats = np.hstack(selected_feats)
  X = np.asfortranarray(X[:, selected_feats])
  Y = np.asfortranarray(Y[:, np.newaxis])
  return X, Y
