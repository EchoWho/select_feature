import matplotlib.pyplot as plt
import numpy as np
import yahoo_common


set_id = 2
show_both=False
partition_id = 0
methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]

orig_whiten = yahoo_common.whiten
yahoo_common.whiten = True
filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id)
d = np.load(filename)
L = d['L']
L = L.item()
b = d['budget']
yahoo_common.whiten = orig_whiten

orig_whiten = yahoo_common.whiten
yahoo_common.whiten = False
filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id)
d_nw = np.load(filename)
L_nw = d_nw['L']
L_nw = L_nw.item()
b_nw = d_nw['budget']
yahoo_common.whiten = orig_whiten

def plot_batch(x, 
               vec_y, 
               names, 
               colors, 
               markers, 
               style,
               score='score',
               fix=True,
               markevery=1) :
  results = [ vec_y[name] for name in names ]
  for name, color, marker, result in zip(names, colors, markers, results) :
    plt.plot(x, result, color=color, linewidth=2, linestyle=style, marker=marker,
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=color, markevery=markevery)


if show_both:
  pass
elif yahoo_common.whiten:
  do_FR = False
  if do_FR :
    plot_batch(b, L, ['FR', 'OMP'], 'rgb', 's+o^', '-')
    #plot_batch(b, L, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b, L, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--')
    plt.legend(('FR - Grouped', 'OMP - Grouped', 'FR - Single', 
                'OMP - Single'), loc='upper right', prop={'size':18})
  else:
    plot_batch(b, L, ['OMP'], 'gb', 's+o^', '-')
    plot_batch(b, L, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b, L, ['OMP SINGLE'], 'gb', '+o^', '--')
    plt.legend(('OMP - Grouped', 'OMP - Grouped Naive',
                'OMP - Single'), loc='upper right', prop={'size':18})

else:
  do_FR = False
  if do_FR :
    plot_batch(b_nw, L_nw, ['FR', 'OMP'], 'rgb', 's+o^', '-')
    #plot_batch(b, L, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b_nw, L_nw, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--')
    plt.legend(('FR - Grouped', 'OMP - Grouped', 'FR - Single', 
                'OMP - Single'), loc='upper right', prop={'size':18})
  else:
    plot_batch(b_nw, L_nw, ['OMP'], 'gb', 's+o^', '-')
    plot_batch(b_nw, L_nw, ['OMP NOINV'], 'g', 'o', ':')
    plot_batch(b_nw, L_nw, ['OMP SINGLE'], 'gb', '+o^', '--')
    plt.legend(('OMP - Grouped', 'OMP - Grouped Naive',
                'OMP - Single'), loc='upper right', prop={'size':18})

plt.show()
