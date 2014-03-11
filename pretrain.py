#!/usr/bin/python
import os,sys
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import yahoo_common

def parse(filename, whiten):
  X, Y = yahoo_common.load_raw_data(filename)
  m_Y = np.mean(Y)
  Y = Y - m_Y
  #Normalize
  m_X = np.zeros(X.shape[1])
  sig_X = np.zeros(X.shape[1])
  L = np.zeros((X.shape[1],X.shape[1]))
  if whiten :
    groups, _ = yahoo_common.load_group()
    for i in range(yahoo_common.n_group_splits) :
      gids = set(groups[i])
      for _, gid in enumerate(gids) :
        selected = [ x_idx for x_idx, g in enumerate(groups[i]) if g==gid ]
        X_group = X[:, selected]
        M = X_group.T.dot(X_group) / X_group.shape[0]
        M = (M + M.T) / 2
        M_diag = np.diag(M)
        eps = 1
        if np.nonzero(M_diag)[0].shape[0] > 0 :
          eps = np.min(M_diag[np.nonzero(M_diag)])
        eps *= 1e-4
        M = M + np.eye(M.shape[0]) * eps 
        L_group = sqrtm(inv(M))
        selected = np.array(selected)
        L[selected[:, np.newaxis], selected] = L_group.real
    
    L = L.T
    X = X.dot(L)
  else:
    m_X = np.mean(X, axis=0)
    sig_X=np.std(X, axis=0)
    sig_X=np.maximum(1, sig_X)
    X= (X - m_X) / sig_X
  # b, C
  b=X.T.dot(Y) / X.shape[0]
  C=X.T.dot(X) / X.shape[0]
  C=(C + C.T) / 2
  return L, m_X, m_Y, sig_X, b, C

#if __name__ == "__main__":
if len(sys.argv)<2:
  print "Usage: python txt2bC.py <set_id>"
else:
  set_id = int(sys.argv[1])
  filename = yahoo_common.filename_data(set_id, 'train')
  L, m_X, m_Y, sig_X, b,C = parse(filename, whiten=yahoo_common.whiten)

  filename = yahoo_common.filename_preprocess_info(set_id)
  np.savez(filename, b=b, C=C, X_std=sig_X, X_mean=m_X, Y_mean=m_Y, L_whiten=L)
