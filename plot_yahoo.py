import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yahoo_common
from yahoo_common import compute_oracle
import sys, os
import bisect 


min_performance = 0.16

def plot_batch(L, 
               x_limit,
               names,
               colors, 
               markers, 
               style,
               score='score',
               fix=True,
               markevery=1) :
  vec_vec_x = [ L[name][0] for name in names ]
  vec_vec_y = [ L[name][1] for name in names ]
  for name, color, marker, vec_x, vec_y in zip(names, colors, markers, vec_vec_x, vec_vec_y) :
    len_limit = bisect.bisect_right(vec_x, x_limit) 
    for i in range(len(vec_y)):
      if vec_y[0] - vec_y[i] >= min_performance:
        break
    plt.plot(vec_x[i:len_limit], 
             (vec_y[0] - vec_y)[i:len_limit], 
             color=color, linewidth=2, linestyle=style, marker=marker,
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=color, markevery=markevery)


def plot_oracle(L, budget_limit):
  colors = ['k', 'm']
  for idx, oracle_str in enumerate(['FR', 'OMP']):
    oracle_costs, oracle_losses = compute_oracle(L[oracle_str][0], L[oracle_str][1])
    nbr_groups = bisect.bisect_right(oracle_costs, budget_limit) 
    for i, loss in enumerate(oracle_losses):
      if oracle_losses[0] - loss >= min_performance:
        break
    plt.plot(oracle_costs[i:nbr_groups], 
             (oracle_losses[0] - oracle_losses)[i:nbr_groups],
             color=colors[idx], linewidth=2, linestyle='-', marker='+',
             markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
             markeredgecolor=colors[idx], markevery=5)


set_id = int(sys.argv[1])
partition_id = 0
group_size = int(sys.argv[2])
l2_lam = 1e-5

group_size2budget_list = dict([(5 , 2478), (10 , 3078), (15, 3898), (20, 2348)] )

# 0 lambda, 1 normalization, 2 cost/no cost, 3 full house
exp_id = 0
if len(sys.argv) > 3:
  exp_id = int(sys.argv[3])
fig = plt.figure(figsize=(10,7), dpi=100)
plt.rc('text', usetex=True)
if exp_id == 0:
  vec_l2_lam = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
  vec_auc = []
  for _, l2_lam in enumerate(vec_l2_lam):
    filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                    group_size, l2_lam, False)
    d = np.load(filename)
    L = d['L']
    L = L.item()
    vec_auc.append(L['OMP'][2])
  plt.plot(vec_l2_lam, vec_auc) 
  plt.xscale('log')

#  plt.xlabel(r"Regularization Constant $\lambda$", fontsize=28)
#  plt.ylabel(r"Reconstruction Error on Validation", fontsize=28)

elif exp_id == 1: #Whiten
  filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                  group_size, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L_no_whiten = L.item()
  filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                  group_size, l2_lam, True)
  d = np.load(filename)
  L = d['L']
  L_whiten = L.item()
  
  budget_limit = group_size2budget_list[group_size]
  nbr_groups = bisect.bisect_right(L_no_whiten['OMP'][0], budget_limit)
  color = 'g'
  start = 3
  plt.plot(L_no_whiten['OMP'][0][start:nbr_groups], 
           (L_no_whiten['OMP'][1][0] - L_no_whiten['OMP'][1])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='s',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  nbr_groups = bisect.bisect_right(L_no_whiten['OMP NOINV'][0], budget_limit)
  color = 'b'
  plt.plot(L_no_whiten['OMP NOINV'][0][start:nbr_groups], 
           (L_no_whiten['OMP NOINV'][1][0] - L_no_whiten['OMP NOINV'][1])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)

#  plot_oracle(L_no_whiten, budget_limit)

  plt.legend(('CS-G-OMP', 'No Whiten'),
             loc='lower right', prop={'size':25})

elif exp_id == 2: # cost / no cost
  filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                  group_size, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L_cost = L.item()
  filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                  group_size, l2_lam, False, True)
  d = np.load(filename)
  L = d['L']
  L_no_cost = L.item()
  
  budget_limit = group_size2budget_list[group_size]
  nbr_groups = bisect.bisect_right(L_cost['OMP'][0], budget_limit) 
  color = 'g'
  start = 0
  print L_cost['OMP'][1][0]
  plt.plot(L_cost['OMP'][0][start:nbr_groups], (L_cost['OMP'][1][0] -  L_cost['OMP'][1])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='s',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)
  nbr_groups = bisect.bisect_right(L_no_cost['OMP'][0], budget_limit) 
  color = 'y'
  plt.plot(L_no_cost['OMP'][0][start:nbr_groups], (L_no_cost['OMP'][1][0] -  L_no_cost['OMP'][1])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)

  #plot_oracle(L_cost, budget_limit)

  plt.legend(('CS-G-OMP', 'G-OMP',), 
             loc='lower right', prop={'size':25})

elif exp_id == 3: # full house
  filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                  group_size, l2_lam, False)
  d = np.load(filename)
  L = d['L']
  L = L.item()
  #budget_limit = group_size2budget_list[group_size] #group_size2budget_list[group_size] / 3.0
  budget_limit = 5000
  plot_oracle(L, budget_limit)
  plot_batch(L, budget_limit, ['FR', 'OMP'], 'rgb', 's+o^', '-', markevery=5)
  plot_batch(L, budget_limit, ['FR SINGLE', 'OMP SINGLE'], 'rgb', '+o^', '--', markevery=5)

  d_lasso = np.load('yahoo_results/spams_%d_%d.npz' % (set_id, group_size))
  nbr_groups = bisect.bisect_right(d_lasso['budget'], budget_limit) 
  color = 'b'
  print L['OMP'][1][0]
  print d_lasso['loss']
  min_performance = 0.15
  for start in range(len(d_lasso['budget'])):
    if L['OMP'][1][0] -  d_lasso['loss'][start] > min_performance:
      break

  plt.plot(d_lasso['budget'][start:nbr_groups], (L['OMP'][1][0] - d_lasso['loss'])[start:nbr_groups],
           color=color, linewidth=2, linestyle='-', marker='o',
           markerfacecolor='none', markersize=7.0, markeredgewidth=1.5,
           markeredgecolor=color)


  plt.legend(('FR Oracle', 'OMP Oracle', 'CS-G-FR', 'CS-G-OMP', 
              'CS-G-FR-Single', 
              'CS-G-OMP-Single', 'Sparse'), loc='lower right', prop={'size':20})
#  plt.xlabel('Feature Cost', fontsize=28)
#  plt.ylabel('Reconstruction Error', fontsize=28)

elif exp_id==4: # Speed up
  methods = ['FR', 'OMP']
  colors='rgb'
  markers='s+o^'
  linestyle = '-'
  filename = yahoo_common.filename_model(set_id, partition_id, 
                                         group_size, l2_lam, False)
  d = np.load(filename)
  for idx, method in enumerate(methods):
    d_method = d[method]
    nbr_groups = bisect.bisect_right(d_method['cost'], 1e8) 
    plt.plot(range(nbr_groups), 
             d_method['time'][:nbr_groups], 
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
  ax.xaxis.set_ticks(np.arange(start, end, (int((end - start ) / 5.1))))
  start, end = ax.get_ylim()
  ax.yaxis.set_ticks(np.arange(start, end, (int((end - start ) / 5.1))))

    
  total_spd_up = 0
  print "%d %d" % (len(d['FR']['time']), len(d['OMP']['time']))
  for i in range(min(len(d['FR']['time']), len(d['OMP']['time']))):
    if i==0:
      continue
    spd_up = d['FR']['time'][i] / d['OMP']['time'][i]
    total_spd_up += spd_up
  total_spd_up /= np.float64(len(d['FR']['time']))
  print total_spd_up
  print d['FR']['time'][-1] / d['OMP']['time'][-1]
  
  plt.savefig('yahoo_results/set%d_size%d_exp%d.png' % (set_id, group_size, exp_id),
              bbox_inches='tight',
              dpi=plt.gcf().dpi)
  
  plt.show()
  exit()
  print "meow"

  

#  plt.xlabel('Number of Feature Groups Selected', fontsize=28)
#  plt.ylabel('Training Time (s)', fontsize=28)


ax = plt.gca()
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, (int((end - start ) / 5.1)) / 100 * 100 ))

def kilos(x, pos):
  if x == 0:
    return '0'
  return '%0.1fK' % (x * 1e-3)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(kilos))

start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, ((end - start ) / 5.1)))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

plt.xlabel('Feature Cost', fontsize=32)
plt.ylabel('Explained Variance', fontsize=32)

plt.savefig('yahoo_results/set%d_size%d_exp%d.png' % (set_id, group_size, exp_id),
            bbox_inches='tight',
            dpi=plt.gcf().dpi)

plt.show()
