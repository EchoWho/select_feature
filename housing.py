import numpy as np
import opt
import os

d = np.load('housing.npz')
X = d['X']
Y = d['Y'].flatten()

args = {}

# if not os.path.isfile('housing_results/optimal.npy'):
#     problem = opt.OptProblem(X, Y, opt.opt_raw_combined, opt.rsquared_combined, opt.gradient, args=args)
#     optimal = opt.alg_optimal(problem, K=13)
#     np.save('housing_results/optimal.npy', optimal)
# else:
#     optimal = np.load('housing_results/optimal.npy')

# uniform_results = opt.all_results(X, Y, optimal=optimal, **args)
# np.savez('housing_results/uniform.npz', **uniform_results)

# d = np.load('housing.costs.npz')
# all_costs = d['all_costs']

# for i,c in enumerate(all_costs):
#     cost_results = opt.all_results(X, Y, optimal=optimal, costs=c, **args)
#     np.savez('housing_results/cost.%d.npz' % i, **cost_results)

lambdas = np.array([0.0, 0.1, 0.25, 0.5])
for i,l in enumerate(lambdas):
     reg_results = opt.all_results(X, Y, optimal=True, l2_lam=l)
     np.savez('housing_results/lambda.%d.npz' % i, **reg_results)

epsilons = np.array([0.3, 0.4, 0.5, 1.0])
for i,e in enumerate(epsilons):
     cons_results = opt.all_results(X, Y, optimal=True, l2_eps=e)
     np.savez('housing_results/epsilon.%d.npz' % i, **cons_results)
