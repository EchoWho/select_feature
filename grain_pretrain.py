#!/usr/bin/python
import os,sys
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import grain_common

def pretrain_state_0(cnt, m_Y, m_X, std_X):
  """
    count, sum_Y, sum_X, sum_X^2, x*y, outer(x,x)
  """
  return (cnt,
          m_Y,
          m_X,
          std_X,
          np.zeros(grain_common.feat_dim), 
          np.zeros((grain_common.feat_dim, grain_common.feat_dim)))

def pretrain_update(state, y, x):
  x = np.array(x)
  state = list(state)
  m_Y = state[1]
  m_X = state[2]
  std_X = state[3]
  x = (x - m_X) / std_X
  state[4] += x * (y - m_Y)
  state[5] += np.outer(x, x)
  return tuple(state)

def pretrain_finalize(state):
  """
    m_Y, m_X, std_X, b, C, L
  """
  cnt = np.float64(state[0])
  b = state[4] / cnt
  C = state[5] / cnt
  C = (C + C.T) / 2.0
  return b, C 

def mean_std_state_0():
  """
    cnt, sum_Y, sum_X, sum_X^2
  """
  return (0, 0, np.zeros(grain_common.feat_dim), np.zeros(grain_common.feat_dim))

def mean_std_update(state, y, x):
  state = list(state)
  x = np.array(x)
  state[0] += 1
  state[1] += y
  state[2] += x
  state[3] += x**2
  return tuple(state)

def mean_std_finalize(state):
  cnt = np.float64(state[0])
  m_Y = state[1] / cnt
  m_X = state[2] / cnt
  std_X = np.sqrt(state[3] / cnt - m_X ** 2)
  std_X += std_X == 0
  return cnt, m_Y, m_X, std_X

def parse(filenames, whiten):
  state = mean_std_state_0()
  for _, filename in enumerate(filenames):
    state = grain_common.load_and_process(filename,
                                          state,
                                          mean_std_update)
  cnt, m_Y, m_X, std_X = mean_std_finalize(state)

  state = pretrain_state_0(cnt, m_Y, m_X, std_X)
  for _, filename in enumerate(filenames):
    state = grain_common.load_and_process(filename,
                                          state,
                                          pretrain_update)
  b, C = pretrain_finalize(state)

  check = False
  if check:
    X, Y = grain_common.load_raw_data(filename)
    cnt2 = X.shape[0]
    m_Y2 = np.mean(Y)
    m_X2 = np.mean(X, axis=0)
    std_X2 = np.std(X, axis=0)
    std_X2 += std_X2 == 0
    X = (X - m_X2) / std_X2
    Y = Y - m_Y2
    b2 = X.T.dot(Y) / X.shape[0]
    C2 = X.T.dot(X) / X.shape[0]
    C2 = (C2 + C2.T) / 2.0

    print np.linalg.norm(b2 - b)
    print np.linalg.norm(C2 - C)

  # Whiten all feature
  L = np.zeros((grain_common.feat_dim, grain_common.feat_dim))
  if whiten :
    d, V = np.linalg.eigh(C)
    d = np.maximum(d, 0)
    D = np.diag(1 / np.sqrt((d + 1e-18)))
    L = np.dot(V, np.dot(D, V.T))

    b = np.dot(L.T, b)
    C = np.dot(L.T, np.dot(C, L))

  return L, m_X, m_Y, std_X, b, C

#if __name__ == "__main__":
if len(sys.argv)<2:
  print "Usage: python txt2bC.py <set_id>"
else:
  set_id = int(sys.argv[1])
  filenames = []
  for s in range(grain_common.nbr_train_sets):
    if (s+1 != set_id):
      filename = grain_common.filename_data(s+1, 'train')
      filenames.append(filename)

  L, m_X, m_Y, std_X, b,C = parse(filenames, whiten=grain_common.whiten)

  filename = grain_common.filename_preprocess_info(set_id)
  np.savez(filename, b=b, C=C, X_std=std_X, X_mean=m_X, Y_mean=m_Y, L_whiten=L)
