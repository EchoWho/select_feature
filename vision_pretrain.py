#!/usr/bin/python
import os,sys
import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import inv
import vision_common

def parse(filename, whiten):
  X, Y = vision_common.load_raw_data(filename)
  m_Y = np.mean(Y)
  print Y.shape
  print X.shape
  Y = Y - m_Y
  #Normalize
  m_X = np.zeros(X.shape[1])
  sig_X = np.zeros(X.shape[1])
  L = 0
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
if __name__ == "__main__":
  filename = vision_common.filename_data('train')
  L, m_X, m_Y, sig_X, b,C = parse(filename, whiten=vision_common.whiten)

  filename = vision_common.filename_preprocess_info()
  np.savez(filename, b=b, C=C, X_std=sig_X, X_mean=m_X, Y_mean=m_Y, L_whiten=L)
