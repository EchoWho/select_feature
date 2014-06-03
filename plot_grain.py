import matplotlib.pyplot as plt
import numpy as np
import grain_common
from yahoo_common import compute_oracle
import matplotlib.ticker as ticker
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
  budget_limit = 0.056
  vec_vec_x = [ L[name][0] for name in names]
  vec_vec_y = [ L[name][y_idx] for name in names]
  for name, color, marker, vec_x, vec_y in zip(names, colors, markers, vec_vec_x, vec_vec_y) :
    nbr_groups = bisect.bisect_right(vec_x, budget_limit) 
    for i in range(len(vec_y)):
      if vec_y[i] < min_performance:
        break
    plt.plot(vec_x[3:nbr_groups], 
             (vec_y[0] - vec_y)[3:nbr_groups],
             color=color, linewidth=2, linestyle=style, 
             marker=marker, markerfacecolor='none', markersize=7.0, markeredgewidth=1.5, markeredgecolor=color, 
             markevery=markevery)

def plot_oracle(L, budget_limit):
  min_performance = 0.065
  budget_limit=0.056
  colors = ['k', 'm']
  for idx, oracle_str in enumerate(['FR', 'OMP']):
    oracle_costs, oracle_losses = compute_oracle(L[oracle_str][0], L[oracle_str][1])
    nbr_groups = bisect.bisect_right(oracle_costs, budget_limit) 
    color = 'k'
    for i in range(len(oracle_losses)):
      if oracle_losses[i] < min_performance:
        break
    plt.plot(oracle_costs[3:nbr_groups], 
             (oracle_losses[0] - oracle_losses)[3:nbr_groups],
             color=colors[idx], linewidth=2, linestyle='-', 
             marker='+', markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=colors[idx], markevery=5)

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
budget_limit = 0.056912
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
  
  
  #plot_oracle(L_no_whiten, 1e8)
  color = 'g'
  start = 2
  nbr_groups = bisect.bisect_right(L_no_whiten['OMP'][0], budget_limit)
  print L_no_whiten['OMP'][y_idx][0]
  plt.plot(L_no_whiten['OMP'][0][start:nbr_groups], 
           (L_no_whiten['OMP'][y_idx][0] - L_no_whiten['OMP'][y_idx])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='s',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  color = 'b'
  nbr_groups = bisect.bisect_right(L_no_whiten['OMP NOINV'][0], budget_limit)
  plt.plot(L_no_whiten['OMP NOINV'][0][start:nbr_groups], 
           (L_no_whiten['OMP NOINV'][y_idx][0] - L_no_whiten['OMP NOINV'][y_idx])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)

  #color = 'b'
  #plt.plot(L_whiten['OMP'][0], L_whiten['OMP'][y_idx],
  #         color=color, linewidth=2, linestyle='-', marker='+',
  #         markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
  #         markeredgecolor=color)
  plt.legend(('CS-G-OMP', 'No-Whiten'),
             loc='lower right', prop={'size':25})

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
  plot_oracle(L, budget_limit)
  plot_batch(L, ['FR', 'OMP'], y_idx, 'rgb', 's+o^', '-', markevery=5)
  plot_batch(L, ['FR SINGLE', 'OMP SINGLE'], y_idx, 'rgb', '+o^', '--', markevery=5)
  
  d_lasso = np.load('grain_results/spams_1.npz')
  nbr_groups = bisect.bisect_right(d_lasso['budget'], budget_limit)
  color = 'b'
  plt.plot(d_lasso['budget'], L['OMP'][1][0] - d_lasso['loss'],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  plt.legend(('FR Oracle', 'OMP Oracle', 'CS-G-FR', 'CS-G-OMP', 
              'CS-G-FR-Single', 
              'CS-G-OMP-Single', 'Sparse'), loc='lower right', prop={'size':20})
#  plt.xlabel('Feature Cost', fontsize=28)
#  plt.ylabel('%s' % (y_label_str), fontsize=28)

elif exp_id==4: # Speed up
  methods = ['FR', 'OMP']
  filename = grain_common.filename_model(set_id, l2_lam, False)
  d = np.load(filename)
  colors='rgb'
  markers='s+o^'
  linestyle = '-'

  for idx, method in enumerate(methods):
    d_method = d[method]
    plt.plot(range(len(d_method['time'])), 
             d_method['time'], 
             color=colors[idx], linewidth=2, linestyle=linestyle, marker=markers[idx],
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=colors[idx])
  plt.legend(('CS-G-FR', 'CS-G-OMP'), loc='upper left', prop={'size':25})
  plt.xlabel('Number of Selected Groups', fontsize=32)
  plt.ylabel('Training Time (sec)', fontsize=32)
  ax = plt.gca()
  ax.tick_params(axis='x', labelsize=30)
  ax.tick_params(axis='y', labelsize=30)
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(start, end, int((end-start) / 5.1)))
  start, end = ax.get_ylim()
  ax.yaxis.set_ticks(np.arange(start, end, int((end-start) / 5.1)))

  total_spd_up = 0
  for i in range(len(d['FR']['time'])):
    if i==0:
      continue
    spd_up = d['FR']['time'][i] / d['OMP']['time'][i]
    total_spd_up += spd_up
  total_spd_up /= np.float64(len(d['FR']['time']))
  print total_spd_up
  plt.savefig('grain_results/set%d_exp%d.png' % (set_id, exp_id),
              bbox_inches='tight',
              dpi=plt.gcf().dpi)

  print d['FR']['time'][-1] / d['OMP']['time'][-1]
  
  plt.show()
  exit()

#  plt.xlabel('Number of Feature Groups Selected', fontsize=28)
#  plt.ylabel('Training Time (s)', fontsize=28)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, (end-start) / 5.1))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

def mss(x, pos):
  if x == 0:
    return '0'
  return '%0.1fK' % (x * 1e-3)

#ax.xaxis.set_major_formatter(ticker.FuncFormatter(kilos))

start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, (end-start) / 5.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

plt.xlabel('Feature Cost', fontsize=32)
plt.ylabel('Explained Variance', fontsize=32)

plt.savefig('grain_results/set%d_exp%d.png' % (set_id, exp_id), 
            bbox_inches='tight',
            dpi=plt.gcf().dpi)
plt.show()
