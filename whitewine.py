import numpy as np
import opt
import os

d = np.load('whitewine.npz')
X = d['X']
Y = d['Y'].flatten()

args = {}

if not os.path.isfile('whitewine_results/optimal.npy'):
    problem = opt.OptProblem(X, Y, opt.opt_raw_combined, opt.rsquared_combined, opt.gradient, args=args)
    optimal = opt.alg_optimal(problem, K=11)
    np.save('whitewine_results/optimal.npy', optimal)
else:
    optimal = np.load('whitewine_results/optimal.npy')

uniform_results = opt.all_results(X, Y, optimal=optimal, **args)
np.savez('whitewine_results/uniform.npz', **uniform_results)

d = np.load('whitewine.costs.npz')
all_costs = d['all_costs']

for i,c in enumerate(all_costs):
    cost_results = opt.all_results(X, Y, optimal=optimal, costs=c, **args)
    np.savez('whitewine_results/cost.%d.npz' % i, **cost_results)
