import numpy as np
import opt
import os

args = {}

# lambdas = np.array([0.0, 0.1, 0.25, 0.5])
# for i in range(20):
#      d = np.load('synthetic/data.%d.npz' % i)
#      X = d['X']
#      Y = d['Y'].flatten()

#      for j,l in enumerate(lambdas):
#           reg_results = opt.all_results_no_cplex(X, Y, K=8, optimal=True, l2_lam=l)
#           np.savez('synthetic/lambda.%d.%d.npz' % (i,j), **reg_results)

for i in range(20):
     d = np.load('synthetic/group.data.%d.npz' % i)
     X = d['X']
     Y = d['Y'].flatten()
     G = d['G']

     group_results = opt.all_results_no_cplex(X, Y, groups=G, optimal=True, l2_lam=1e-4)
     np.savez('synthetic/group.%d.npz' % i, **group_results)

# lambdas = np.array([0.0, 0.1, 0.25, 0.5, 0.2, 0.3, 0.4])
# for i in range(20):
#      d = np.load('synthetic/data.%d.npz' % i)
#      X = d['X']
#      Y = d['Y'].flatten()

#      for j,l in enumerate(lambdas):
#           args['l2_lam'] = l
#           p = opt.OptProblem(X, Y, opt.opt_raw_combined, opt.rsquared_combined_bC, opt.gradient_bC, args=args)

#           f = 'synthetic/lambda.%d.%d.npz' % (i,j)
#           d = np.load(f)
#           reg_results = opt.fix_results_p(d, p, K=30, l1=True)
#           np.savez(f, **reg_results)
