import matplotlib.pyplot as plt
import numpy as np
import grain_common
import sys,os


#set_id = grain_common.default_set_id
set_id = int(sys.argv[1])
show_both=False
plot_err = True

partition_id = 0
methods = ['OMP', 'OMP NOINV', 'OMP SINGLE', 'FR', 'FR SINGLE' ]

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

opt_comparison = False
costs_vs_no_costs = True
if opt_comparison:
  orig_whiten = grain_common.whiten
  grain_common.whiten = False
  filename = grain_common.filename_budget_vs_loss(set_id)
  d_nw = np.load(filename)
  if plot_err:
    L_nw = d_nw['err']
  else:
    L_nw = d_nw['L']
  L_nw = L_nw.item()
  costs_nw = d_nw['costs']
  costs_nw = costs_nw.item()
  grain_common.whiten = orig_whiten

  plot_batch(costs_nw, L_nw, ['FR', 'OMP'], 'rgb', 's+o^', '-')
  plot_batch(costs_nw, L_nw, ['OMP NOINV'], 'g', 'o', ':')
  plot_batch(costs_nw, L_nw, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--')
  plt.legend(('FR - Grouped', 'OMP - Grouped', 'OMP - Grouped Naive', 'FR - Single', 
              'OMP - Single'), loc='upper right', prop={'size':18})
  plt.xlabel('Total Feature Cost', fontsize=28)
  if plot_err:
    plt.ylabel('Error Rate', fontsize=28)
  else:
    plt.ylabel('Squared Loss', fontsize=28)
  plt.gca().tick_params(axis='both', which='major', labelsize=15)
  plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
  fig = plt.gcf()
  fig.savefig(grain_common.filename_budget_vs_loss_img(set_id, plot_err, grain_common.ignore_costs))

elif costs_vs_no_costs:
  filenames = []
  orig_whiten = grain_common.whiten
  orig_ignore_costs = grain_common.ignore_costs
  grain_common.whiten = False
  grain_common.ignore_costs = False
  filenames.append(grain_common.filename_budget_vs_loss(set_id))
  grain_common.ignore_costs = True
  filenames.append(grain_common.filename_budget_vs_loss(set_id))
  grain_common.whiten = orig_whiten
  grain_common.ignore_costs = orig_ignore_costs

  d = []
  L = []
  costs = []
  colors = [ 'g', 'b'] 
  if plot_err:
    L_name = 'err'
  else:
    L_name = 'L'
  for i in range(2):
    d.append(np.load(filenames[i]))
    l = d[-1][L_name]
    L.append(l.item())
    c = d[-1]['costs']
    costs.append(c.item())
    plot_batch(costs[-1], L[-1], ['OMP'], colors[i], '+o^', '-')

  plt.legend(('OMP w/ costs', 'OMP w/o costs'), loc='upper right', prop={'size':18})
  plt.xlabel('Total Feature Cost', fontsize=28)
  if plot_err:
    plt.ylabel('Error Rate', fontsize=28)
  else:
    plt.ylabel('Squared Loss', fontsize=28)
  plt.gca().tick_params(axis='both', which='major', labelsize=15)
  plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
  fig = plt.gcf()
  fig.savefig('./grain_results/budget_vs_loss/%s_costs_vs_no_costs.png' % (L_name))

plt.show()
