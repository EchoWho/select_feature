import numpy as np
import opt
import os,sys
import vision_common

test_i = int(sys.argv[1])
bov_files = [ 'ca01_no_label_bov.mat', 
    'ca02_no_label_bov.mat', 
    'ca03_no_label_bov.mat', 
    'ca04_no_label_bov.mat', 
    'ca05_no_label_bov.mat', 
    'ca06_no_label_bov.mat', 
    'ca07_no_label_bov.mat', 
    'ca08_no_label_bov.mat', 
    'ca09_no_label_bov.mat', 
    'ca10_no_label_bov.mat', 
    'ca11_no_label_bov.mat'] 

X = [] 
Y = []
for i in range(len(bov_files)):
  if i != test_i:
    X_i, Y_i = vision_common.load_raw_data('%s/%s' % (vision_common.data_dir, bov_files[i]))
    X.extend(X_i)
    Y.extend(Y_i)
X = np.array(X)
Y = np.array(Y)
X_mean = np.mean(X, axis = 0)
X_std = np.std(X, axis = 0)
X_std += X_std == 0
X = (X  - X_mean) / X_std
Y_mean = np.mean(Y, axis = 0)
Y = Y - Y_mean

b=X.T.dot(Y) / X.shape[0]
C=X.T.dot(X) / X.shape[0]
C=(C + C.T) / 2

np.savez(vision_common.filename_preprocess_info(test_i), b=b, C=C, 
      X_std=X_std, X_mean=X_mean, Y_mean=Y_mean, L_whiten=0)

l2_lam = 1e-6
C = C + np.eye(C.shape[0]) * l2_lam

args = {}

# uniform_results = opt.all_results_bC(b, C, **args)
# np.savez('yahoo_results_small/uniform.npz', **uniform_results)

# c = np.load('yahoo.costs.npy').astype(float)

# cost_results = opt.all_results_bC(b, C, costs=c, **args)
# np.savez('yahoo_results_small/cost.time.npz', **cost_results)

groups, costs = vision_common.load_group()
group_results = opt.all_results_bC(b, C, costs=costs, groups=groups, optimal=False)
filename = vision_common.filename_model(test_i, l2_lam)
np.savez(filename, **group_results)
