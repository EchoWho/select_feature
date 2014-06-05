import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import yahoo_common

from plot_common import *


set_id = 1
partition_id = 0
group_size = 10
l2_lam = 1e-5

d = np.load(yahoo_common.filename_model(set_id, partition_id, 
                                              group_size, l2_lam, False))
d = d['OMP']
score = d['score'][0:57]
cost = d['cost'][0:57]
#return cost[np.sum( score <= alpha_score ) - 1]


plt.hold(True)
alphas = [0.9, 0.95, 0.98, 0.99]
for _, alpha in enumerate(alphas):
  stopping_idx = np.sum( score <= alpha * score[-1])
  stopping_cost = cost[stopping_idx]

  print "alpha %f ; stopping_cost %f" % (alpha, stopping_cost)
  
  vert_line = plt.plot([stopping_cost, stopping_cost], [0, score[stopping_idx]], linewidth=2) 

plt.plot(cost[3:], score[3:], 
         linewidth=2)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=tickfontsize())
ax.tick_params(axis='y', labelsize=tickfontsize())


def kilos(x, pos):
  if x == 0:
    return '0'
  return '%dK' % (x * 1e-3)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(kilos))
start, end = ax.get_xlim()
plt.xticks(np.arange(start, end, 5000))

start, end = ax.get_ylim()
plt.yticks(np.arange(start, end, (end - start) / 5.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

plt.xlabel('Feature Cost', fontsize=labelfontsize()) 
plt.ylabel('Explained Variance', fontsize=labelfontsize())

plt.rc('text', usetex=True)
plt.legend((r"$\alpha$ = 0.9",
            r"$\alpha$ = 0.95",
            r"$\alpha$ = 0.98",
            r"$\alpha$ = 0.99"), loc = 'lower right', prop={'size':25}  )


plt.savefig('/home/hanzhang/projects/select_feature/paper/img/timeliness.png',
            bbox_inches='tight', dpi = plt.gcf().dpi)
plt.show()
  
