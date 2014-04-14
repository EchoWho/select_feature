import matplotlib.pyplot as plt
import numpy as np
import grain_common


set_id = grain_common.default_set_id
show_both=False
partition_id = 0
methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]

"""
orig_whiten = grain_common.whiten
grain_common.whiten = True
filename = grain_common.filename_budget_vs_loss(set_id, partition_id)
d = np.load(filename)
L = d['L']
L = L.item()
b = d['budget']
grain_common.whiten = orig_whiten
"""

orig_whiten = grain_common.whiten
grain_common.whiten = False
filename = grain_common.filename_budget_vs_loss(set_id)
d_nw = np.load(filename)
L_nw = d_nw['L']
L_nw = L_nw.item()
costs_nw = d_nw['costs']
costs_nw = costs_nw.item()
grain_common.whiten = orig_whiten

def plot_batch(vec_x, 
               vec_y, 
               names, 
               colors, 
               markers, 
               style,
               score='score',
               fix=True,
               markevery=1) :
  costs = [ vec_x[name] for name in names ]
  results = [ vec_y[name] for name in names ]
  for name, color, marker, result, cost in zip(names, colors, markers, results, costs) :
    plt.plot(cost[:25], result[:25], color=color, linewidth=2, linestyle=style, marker=marker,
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=color, markevery=markevery)


do_FR = True
if show_both:
  pass
elif grain_common.whiten:
  if do_FR :
    plot_batch(b, L, ['FR', 'OMP'], 'rgb', 's+o^', '-')
    plot_batch(b, L, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b, L, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--')
    plt.legend(('FR - Grouped', 'OMP - Grouped', 'OMP - Grouped Naive', 'FR - Single', 
                'OMP - Single'), loc='upper right', prop={'size':18})
  else:
    plot_batch(b, L, ['OMP'], 'gb', 's+o^', '-')
    plot_batch(b, L, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b, L, ['OMP SINGLE'], 'gb', '+o^', '--')
    plt.legend(('OMP - Grouped', 'OMP - Grouped Naive',
                'OMP - Single'), loc='upper right', prop={'size':18})
else:
  if do_FR :
    plot_batch(costs_nw, L_nw, ['FR', 'OMP'], 'rgb', 's+o^', '-')
    plot_batch(costs_nw, L_nw, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(costs_nw, L_nw, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--')
    plt.legend(('FR - Grouped', 'OMP - Grouped', 'OMP - Grouped Naive', 'FR - Single', 
                'OMP - Single'), loc='upper right', prop={'size':18})
  else:
    plot_batch(costs_nw, L_nw, ['OMP'], 'gb', 's+o^', '-')
    plot_batch(costs_nw, L_nw, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(costs_nw, L_nw, ['OMP SINGLE'], 'gb', '+o^', '--')
    plt.legend(('OMP - Grouped', 'OMP - Grouped Naive',
                'OMP - Single'), loc='upper right', prop={'size':18})

plt.show()
