import numpy as np

n_group_splits=1
feat_dim=700
whiten=False

root_dir='/home/hanzhang/code/speedboost/select_feature'
data_dir='%s/yahoo_data' % (root_dir)
result_dir='%s/yahoo_results' % (root_dir)

def param_str() :
  if not whiten :
    return 'no_whiten.'
  return ''

def filename_data(set_id, mode) :
  """ Training data filename """
  return '%s/set%d.%s.txt' % (data_dir, set_id, mode)

def filename_preprocess_info(set_id) : 
  """ 
    Filename containing information needed for preprocess data, 
    and (b, C) for training 
  """
  return '%s/yahoo.set%d.train.bC.%snpz' % (data_dir, set_id, param_str())
 
def filename_model(set_id, partition_id) :
  """ Filename for model trained by feature selection """
  return '%s/set%d.model.group%d.%snpz' % (result_dir, set_id, partition_id, param_str())

def filename_budget_vs_loss(set_id, partition_id) :
  """ Filenmae for final results """
  return '%s/set%d.budget_vs_loss.group%d.%snpz' % (result_dir, set_id, partition_id, param_str())

def preprocess_X(X_raw, set_id) :
  filename = filename_preprocess_info(set_id)
  d = np.load(filename)
  if whiten :
    L_whiten = d['L_whiten']
    X = X_raw.dot(L_whiten)
  else :
    X_mean = d['X_mean']
    X_std = d['X_std']
    #X = (X_raw - X_mean) / X_std
  d.close()
  return X

def load_group() :
  d_groups = np.load('%s/yahoo.groups.npz' % (data_dir))
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
