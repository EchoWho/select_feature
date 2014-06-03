import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import yahoo_common
import grain_common

from plot_common import *


set_id = 1
partition_id = 0
group_size = 10
l2_lam = 1e-7

d = np.load(grain_common.filename_model(set_id, l2_lam, False))
d = d['OMP']
score = d['score']
cost = d['cost']
final_idx = np.sum( cost < 1e8 ) - 1
final_score = score[final_idx]
#return cost[np.sum( score <= alpha_score ) - 1]


plt.hold(True)
alphas = [0.9, 0.95, 0.97, 0.99]
for _, alpha in enumerate(alphas):
  stopping_idx = np.sum( score <= alpha * final_score) - 1
  stopping_cost = cost[stopping_idx]

  print "alpha %f ; stopping_cost %f" % (alpha, stopping_cost)
  
  vert_line = plt.plot([stopping_cost, stopping_cost], [0, score[stopping_idx]], linewidth=2) 

plt.plot(cost[:(final_idx + 1)], score[:(final_idx + 1)], 
         linewidth=2)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=tickfontsize())
ax.tick_params(axis='y', labelsize=tickfontsize())


def kilos(x, pos):
  if x == 0:
    return '0'
  return '%dK' % (x * 1e-3)

ax.xaxis.set_major_formatter(FuncFormatter(kilos))
plt.xticks(np.arange(0.0, 18000, 5000))
plt.yticks(np.arange(0.0, 0.36, 0.08))

plt.xlabel('Feature Cost', fontsize=labelfontsize()) 
plt.ylabel('Explained Variance', fontsize=labelfontsize())

plt.rc('text', usetex=True)
plt.legend((r"$\alpha$ = 0.9",
            r"$\alpha$ = 0.95",
            r"$\alpha$ = 0.97",
            r"$\alpha$ = 0.99"), loc = 'lower right', prop={'size':25}  )


plt.savefig('/home/hanzhang/projects/select_feature/paper/img/timeliness.png',
            bbox_inches='tight', dpi = plt.gcf().dpi)
#plt.show()
  
