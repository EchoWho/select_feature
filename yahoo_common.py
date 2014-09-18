import numpy as np
import scipy.sparse as ssp

n_group_splits=1
feat_dim=700
whiten=False

root_dir='/home/hanzhang/code/speedboost/select_feature'
data_dir='%s/yahoo_data' % (root_dir)
result_dir='%s/yahoo_results' % (root_dir)

def param_str(do_whiten=whiten, ignore_cost=False) :
  s = ''
  if not do_whiten :
    s += 'no_whiten.'
  if ignore_cost :
    s += 'eq_costs.'
  return s

def filename_data(set_id, mode) :
  """ Training data filename """
  return '%s/set%d.%s.svmlight' % (data_dir, set_id, mode)

def filename_preprocess_info(set_id, do_whiten=whiten) : 
  """ 
    Filename containing information needed for preprocess data, 
    and (b, C) for training 
  """
  return '%s/yahoo.set%d.train.bC.%snpz' % (data_dir, set_id, param_str(do_whiten))
 
def filename_model(set_id, partition_id, group_size=5, l2_lam=1e-6, do_whiten=whiten,
  ignore_cost=False) :
  """ Filename for model trained by feature selection """
  return '%s/set%d.model.group%d.size%d.lam%f.%snpz' % (result_dir, 
    set_id, partition_id, group_size, l2_lam, param_str(do_whiten, ignore_cost))

def filename_budget_vs_loss(set_id, partition_id, group_size=5, l2_lam=1e-6, do_whiten=whiten,
  ignore_cost=False) :
  """ Filenmae for final results """
  return '%s/set%d.budget_vs_loss.group%d.size%d.lam%f.%snpz' % (result_dir, 
    set_id, partition_id, group_size, l2_lam, param_str(do_whiten, ignore_cost))

def preprocess_X(X_raw, Y_raw, set_id, do_whiten=whiten) :
  filename = filename_preprocess_info(set_id, do_whiten)
  d = np.load(filename)
  if whiten :
    L_whiten = d['L_whiten']
    X_mean = d['X_mean']
    X = np.dot(X_raw - X_mean, L_whiten)
  else :
    X_mean = d['X_mean']
    X_std = d['X_std']
    X = (X_raw - X_mean) / X_std
  Y = Y_raw - d['Y_mean']
  d.close()
  return X, Y

def preprocess_X_rebut(X_raw, Y_raw, set_id, do_whiten=whiten):
  filename = filename_preprocess_info(set_id, do_whiten)
  d = np.load(filename)
  if whiten :
    L_whiten = d['L_whiten']
    X_mean = d['X_mean']
    X = np.dot(X_raw - X_mean, L_whiten)
  else :
    X_mean = d['X_mean']
    X_std = d['X_std']
    X = (X_raw - X_mean) / X_std
#  Y = Y_raw - d['Y_mean']
  Y_mean = d['Y_mean']
  d.close()
  return X, Y_mean

def load_group(group_size=10) :
  d_groups = np.load('%s/yahoo.groups.size.%d.npz' % (data_dir, group_size))
  groups = d_groups['groups']
  costs = d_groups['costs']
  d_groups.close()
  return groups, costs

def load_raw_data(filename) :
  X = []
  Y = []
  fin = open(filename, 'r')
  for l in fin :
    features = l.rstrip().split(' ')
    dataline = [0] * feat_dim
    for (i, f) in enumerate(features) :
      if i == 0 :
        Y.append(int(f))
      else:
        indfeat = f.split(':')
        if indfeat[0] != 'qid' :
          dataline[int(indfeat[0])-1] = np.float64(indfeat[1])
    X.append(dataline)
  X = np.array(X)
  Y = np.array(Y)
  fin.close()
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

def create_spams_params(groups, costs):
  spams_params = {'numThreads' : -1,'verbose' : True,
         'lambda1' : 0.001, 'it0' : 10, 'max_it' : 500,
         'L0' : 0.1, 'tol' : 1e-5, 'intercept' : False,
         'pos' : False}
  group_names = sorted(list(set(groups)))
  nbr_groups = len(group_names)
  group_sizes = [ len(np.nonzero(groups == g)[0]) for g in group_names ]
  eta_g = [ costs[g] for g in group_names ]
  eta_g = np.array([1e-9] + eta_g)
  group_sizes = [0] + group_sizes
  group_own = np.cumsum([0] + list(group_sizes)[:-1])
  group_own = group_own.astype(np.int32)
  group_sizes = np.array(group_sizes, dtype=np.int32)
  
  spams_groups = np.zeros((nbr_groups + 1, nbr_groups+1), dtype=np.bool)
  spams_groups[1:, 0] = 1
  spams_groups = ssp.csc_matrix(spams_groups, dtype=np.bool)

  spams_tree = {'eta_g' : eta_g , 'groups' : spams_groups, 
    'own_variables' : group_own, 'N_own_variables' : group_sizes }
  spams_params['compute_gram'] = True
  spams_params['loss'] = 'square'
  spams_params['regul'] = 'tree-l2'
  
  return spams_tree, spams_params

def compute_stopping_cost(alpha, d):
  d = d['OMP']
  score = d['score']
  cost = d['cost']
  alpha_score = score[np.sum( cost < 1e8 ) - 1] * alpha
  return cost[np.sum( score <= alpha_score ) - 1]

def compute_auc(costs, losses, stopping_cost):
  auc = 0
  for i in range(len(costs) - 1):
    if costs[i] >= stopping_cost:
      break
    if costs[i + 1] > stopping_cost:
      a = (stopping_cost - costs[i]) / 2.0 / (costs[i+1] - costs[i])
      auc += (losses[i+1] * a   + losses[i] * (1-a) ) * (stopping_cost - costs[i])
    else:
      auc += (costs[i + 1] - costs[i]) * (losses[i+1] + losses[i]) / 2.0
  auc /= stopping_cost * losses[0]
  return 1 - auc

def compute_oracle(costs, losses):
  c = []
  l = []
  for i in range(len(costs) - 1):
    c.append(costs[i + 1] - costs[i])
    l.append(losses[i] - losses[i + 1])
  l = np.array(l)
  c = np.array(c)
  l_over_c = 0.0 - l / c
  sorted_idx = sorted(range(len(l)), key=lambda x : l_over_c[x])
  oracle_costs = [costs[0]]
  oracle_losses = [losses[0]]
  for _, i in enumerate(sorted_idx):
    oracle_costs.append(oracle_costs[-1] + c[i])
    oracle_losses.append(oracle_losses[-1] - l[i])
  return np.array(oracle_costs), np.array(oracle_losses)
