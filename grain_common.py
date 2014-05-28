import numpy as np

n_group_splits=1
default_set_id=4
nbr_train_sets=5
feat_dim=328
whiten=False

root_dir='/home/hanzhang/projects/select_feature/code'
data_dir='%s/grain_data' % (root_dir)
result_dir='%s/grain_results' % (root_dir)

def param_str(do_whiten=whiten, ignore_cost=False) :
  s = ''
  if not do_whiten :
    s += 'no_whiten.'
  if ignore_cost:
    s += 'eq_costs.'
  return s

def filename_data(set_id, mode) :
  """ Training data filename """
  return '%s/set%d.%s.txt' % (data_dir, set_id, mode)

def filename_preprocess_info(set_id, do_whiten=whiten) : 
  """ 
    Filename containing information needed for preprocess data, 
    and (b, C) for training 
  """
  # it's "false" here because we recompute the L at preprocessing time and training time 
  # for now TODO
  return '%s/grain.set%d.train.bC.%snpz' % (data_dir, set_id, param_str(False))
 
def filename_model(set_id, lam, do_whiten=whiten, ignore_cost=False) :
  """ Filename for model trained by feature selection """
  return '%s/set%d.model.lam%f.%snpz' % (result_dir, set_id, lam, 
    param_str(do_whiten, ignore_cost))

def filename_budget_vs_loss(set_id, lam, do_whiten=whiten, ignore_cost=False) :
  """ Filenmae for final results """
  return '%s/set%d.budget_vs_loss.lam%f.%snpz' % (result_dir, set_id, lam, 
    param_str(do_whiten, ignore_cost))

def filename_budget_vs_loss_img(set_id, plot_err, unit_costs) :
  """ Filenmae for final results """
  loss_err = ''
  if plot_err:
    loss_err = 'err'
  else:
    loss_err = 'loss'
  s = ''
  if unit_costs:
    s = 'unit_'
  return '%s/budget_vs_loss/%s_%scost_%d.png' % (result_dir, loss_err, s, set_id)

def preprocess_X(X_raw, Y_raw, set_id) :
  filename = filename_preprocess_info(set_id)
  doc = np.load(filename)
  if whiten :
    C = doc['C']
    d, V = np.linalg.eigh(C)
    d = np.maximum(d, 0)
    D = np.diag(1 / np.sqrt(d + 1e-18))
    L = np.dot(np.dot(V, D), V.T)
    X = X_raw.dot(L)
  else :
    X_mean = doc['X_mean']
    X_std = doc['X_std']
    X_std += X_std==0
    X = (X_raw - X_mean) / X_std
  Y_mean = doc['Y_mean']
  Y = Y_raw - Y_mean
  doc.close()
  return X, Y

def load_group() :
  d_groups = np.load('%s/grain.groups.npz' % (data_dir))
  groups = d_groups['groups']
  costs = d_groups['costs']
  d_groups.close()
  return groups, costs

def load_raw_data(filename) :
  X = []
  Y = []
  fin = open(filename, 'r')
  for l in fin :
    features = l.rstrip().split(',')
    dataline = [0] * feat_dim
    for (i, f) in enumerate(features) :
      if i == 0 :
        Y.append(int(f))
      else:
        dataline[i-1] = np.float64(f)
    X.append(dataline)
  X = np.array(X)
  Y = np.array(Y)
  fin.close()
  return X, Y

def load_and_process(filename, 
                     state_0, 
                     update_func): 
  state = state_0
  fin = open(filename, 'r')
  for l in fin:
    fs = l.rstrip().split(',')
    x = [0] * feat_dim
    for (i, f) in enumerate(fs):
      if i == 0:
        y = int(f)
      else:
        x[i-1] = np.float64(f)
    state = update_func(state, y, x)
  fin.close()
  return state
