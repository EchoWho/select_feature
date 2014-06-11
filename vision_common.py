import numpy as np
import scipy.sparse as ssp
import scipy.io

n_group_splits=1
feat_dim=2400
whiten=False

root_dir='/home/hanzhang/projects/select_feature/code'
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

def filename_preprocess_info(test_i, do_whiten=whiten) : 
  """ 
    Filename containing information needed for preprocess data, 
    and (b, C) for training 
  """
  return '%s/vision.train%d.bC.%snpz' % (data_dir, test_i, param_str(do_whiten))
 
def filename_model(test_i, l2_lam=1e-6, do_whiten=whiten, ignore_cost=False) :
  """ Filename for model trained by feature selection """
  return '%s/model.%d.lam%f.%snpz' % (result_dir, test_i,l2_lam, param_str(do_whiten, ignore_cost))

def filename_budget_vs_loss(test_i, l2_lam=1e-6, do_whiten=whiten, ignore_cost=False) :
  """ Filenmae for final results """
  return '%s/budget_vs_loss.%d.lam%f.%snpz' % (result_dir, test_i, l2_lam, param_str(do_whiten, ignore_cost))

def preprocess_X(X_raw, Y_raw, test_i, do_whiten=whiten) :
  filename = filename_preprocess_info(test_i, do_whiten)
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
  if d_raw.has_key('X'):
    X = d_raw['X']
    Y = d_raw['Y'][0, :]
  else:
    X = d_raw['feats'].T
    Y = d_raw['label'][0,:]
  return X, Y
