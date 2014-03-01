import numpy as np

n_group_splits=1
feat_dim=700
whiten=True

def load_group() :
  d_groups = np.load('data/yahoo.groups.npz')
  groups = d_groups['groups']
  costs = d_groups['costs']
  return groups, costs

def load_raw_data(filename) :
  X = []
  Y = []
  for l in open(filename, 'r') :
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
  return X, Y
