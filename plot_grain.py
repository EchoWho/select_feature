import matplotlib.pyplot as plt
import numpy as np
import grain_common
from yahoo_common import compute_oracle
import sys,os
import bisect

def plot_batch(L, 
               names, 
               y_idx,
               colors, 
               markers, 
               style,
               score='score',
               fix=True,
               markevery=1) :
  min_performance = 0.065
  budget_limit = 0.04
  vec_vec_x = [ L[name][0] for name in names]
  vec_vec_y = [ L[name][y_idx] for name in names]
  for name, color, marker, vec_x, vec_y in zip(names, colors, markers, vec_vec_x, vec_vec_y) :
    nbr_groups = bisect.bisect_right(vec_x, budget_limit) 
    for i in range(len(vec_y)):
      if vec_y[i] < min_performance:
        break
    plt.plot(vec_x[i:nbr_groups], vec_y[i:nbr_groups],
             color=color, linewidth=2, linestyle=style, marker=marker,
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=color, markevery=markevery)

def plot_oracle(L, budget_limit):
  min_performance = 0.065
  budget_limit=0.04
  oracle_costs, oracle_losses = compute_oracle(L['FR'][0], L['FR'][1])
  nbr_groups = bisect.bisect_right(oracle_costs, budget_limit) 
  color = 'k'
  for i in range(len(oracle_losses)):
    if oracle_losses[i] < min_performance:
      break
  plt.plot(oracle_costs[i:nbr_groups], oracle_losses[i:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='+',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)

set_id = int(sys.argv[1])
plot_err = False
if plot_err:
  y_idx = 3
  y_label_str = "Error Rate"
else:
  y_idx = 1
  y_label_str = "Reconstruction Error"
nbr_feat_chosen = 58

exp_id = 0
l2_lam = 1e-7
if len(sys.argv) > 2:
  exp_id = int(sys.argv[2])

fig = plt.figure(figsize=(10,7), dpi=100)
plt.rc('text', usetex=True)
if exp_id == 0:
  vec_l2_lam = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
  vec_auc = []
  for _, l2_lam in enumerate(vec_l2_lam):
    filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False)
    d = np.load(filename)
    L = d['L']
    L = L.item()
    vec_auc.append(L['OMP'][2])
  plt.plot(vec_l2_lam, vec_auc) 
  plt.xscale('log')
#  plt.xlabel(r"Regularization Constant $\lambda$", fontsize=28)
#  plt.ylabel('%s on Validation' % (y_label_str), fontsize=28)

elif exp_id == 1:
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L_no_whiten = L.item()
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, True)
  d = np.load(filename)
  L = d['L']
  L_whiten = L.item()
  
  
  plot_oracle(L_no_whiten, 1e8)
  color = 'r'
  plt.plot(L_no_whiten['OMP'][0], L_no_whiten['OMP'][y_idx],
           color=color, linewidth=2, linestyle='-', marker='s',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  color = 'g'
  plt.plot(L_no_whiten['OMP NOINV'][0], L_no_whiten['OMP NOINV'][y_idx],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)

  #color = 'b'
  #plt.plot(L_whiten['OMP'][0], L_whiten['OMP'][y_idx],
  #         color=color, linewidth=2, linestyle='-', marker='+',
  #         markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
  #         markeredgecolor=color)
  plt.legend(('Oracle', 'Group-Whiten', 'No-Group-Whiten'),
             loc='upper right', prop={'size':25})
#  plt.xlabel('Feature Cost', fontsize=28)
#  plt.ylabel('%s' % (y_label_str), fontsize=28)

elif exp_id==2: # cost / no cost
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L_cost = L.item()
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False, True)
  d = np.load(filename)
  L = d['L']
  L_no_cost = L.item()
  
  color = 'r'
  plt.plot(L_cost['OMP'][0], L_cost['OMP'][y_idx],
           color=color, linewidth=2, linestyle='-', marker='s',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  color = 'g'
  plt.plot(L_no_cost['OMP'][0], L_no_cost['OMP'][y_idx],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  plot_oracle(L_cost, 1e8)
  plt.legend(('Cost Sensitive OMP', 'OMP', 'Oracle'), 
              loc='upper right', prop={'size':25})
#  plt.xlabel('Feature Cost', fontsize=28)
#  plt.ylabel('%s' % (y_label_str), fontsize=28)

elif exp_id == 3:
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L = L.item()
  plot_oracle(L, 1e8)
  plot_batch(L, ['FR', 'OMP'], y_idx, 'rgb', 's+o^', '-')
  plot_batch(L, ['FR SINGLE', 'OMP SINGLE'], y_idx, 'rgb', '+o^', '--')
  plot_batch(L, ['OMP NOINV'], y_idx, 'b', '^', '-')
  filename = grain_common.filename_budget_vs_loss(set_id, l2_lam, False, True)
  d = np.load(filename)
  L_no_cost = d['L']
  L_no_cost = L_no_cost.item()
  plot_batch(L_no_cost, ['OMP'], y_idx, 'y', '^', '-')
  plt.legend(('FR Oracle', 'G-FR', 'G-OMP', 
              'G-FR-Single', 
              'G-OMP-Single', 'G-OMP-No-whiten', 'G-OMP-Ignore-Cost',
              'Oracle'), loc='upper right', prop={'size':20})
  plt.xticks(np.arange(0.0, 0.05, 0.01), rotation=30)
  plt.yticks(np.arange(0.046, 0.064, 0.004), rotation=30)
#  plt.xlabel('Feature Cost', fontsize=28)
#  plt.ylabel('%s' % (y_label_str), fontsize=28)

elif exp_id==4: # Speed up
  methods = ['OMP', 'OMP SINGLE', 'FR', 'FR SINGLE']
  filename = grain_common.filename_model(set_id, l2_lam, False)
  d = np.load(filename)
  L = []
  for _, method in enumerate(methods):
    d_method = d[method]
    L.append((range(d_method['time'].shape[0]), d_method['time']))
  L = dict(zip(methods, L))
  plot_batch(L, ['FR', 'OMP'], 1, 'rgb', 's+o^', '-')
  #plot_batch(L, ['FR SINGLE', 'OMP SINGLE'], 1, 'rgb', '+o^', '--')
  plt.legend(('FR - Grouped', 'OMP - Grouped'), loc='upper left', prop={'size':25})

  total_spd_up = 0
  for i in range(len(d['FR']['time'])):
    if i==0:
      continue
    spd_up = d['FR']['time'][i] / d['OMP']['time'][i]
    total_spd_up += spd_up
  total_spd_up /= np.float64(len(d['FR']['time']))
  print total_spd_up
#  plt.xlabel('Number of Feature Groups Selected', fontsize=28)
#  plt.ylabel('Training Time (s)', fontsize=28)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=33)
ax.tick_params(axis='y', labelsize=33)

plt.savefig('grain_results/set%d_exp%d.png' % (set_id, exp_id), 
            bbox_inches='tight',
            dpi=plt.gcf().dpi)
plt.show()
