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
    m_X = np.mean(X, axis=0)
    X = X - m_X
    C = np.dot(X.T, X) / X.shape[0]
    C = (C + C.T) / 2.0
    d, V = np.linalg.eigh(C)
    d = np.maximum(d, 0.0)
    D = np.diag(1.0 / np.sqrt(d + 1e-18))
    L = np.dot(np.dot(V, D), V.T)
    X = np.dot(X, L)
  else:
    m_X = np.mean(X, axis=0)
    sig_X=np.std(X, axis=0)
    sig_X += sig_X ==0
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
