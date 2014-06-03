import matplotlib.pyplot as plt
import numpy as np
import yahoo_common
from yahoo_common import compute_auc, compute_oracle
import sys, os
import bisect 


group_size = int(sys.argv[1])
exp_id = int(sys.argv[2])

vec_set_ids = [1]

partition_id = 0
l2_lam = 1e-5

#Compute stopping cost
alpha = 0.975
oracle_str = 'FR'
if exp_id > 5:
  print "change oracle str to OMP and perfrom exp id 1"
  exp_id = 1
  oracle_str = 'OMP'

if exp_id==0:
  pass
elif exp_id == 1:
  auc = dict([ ('OMP', 0), ('OMP NOINV', 0), ('Oracle', 0) ])
  for _, set_id in enumerate(vec_set_ids):
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

    d_model = np.load(yahoo_common.filename_model(set_id, partition_id, 
                                                  group_size, l2_lam, False))
    stopping_cost = yahoo_common.compute_stopping_cost(alpha, d_model)

    oracle_costs, oracle_losses = compute_oracle(L_no_whiten[oracle_str][0], L_no_whiten[oracle_str][1])
    auc['Oracle'] += compute_auc(oracle_costs, oracle_losses, stopping_cost)
    auc['OMP'] += compute_auc(L_no_whiten['OMP'][0], L_no_whiten['OMP'][1],
                              stopping_cost)
    auc['OMP NOINV'] += compute_auc(L_no_whiten['OMP NOINV'][0], 
                                    L_no_whiten['OMP NOINV'][1],
                                    stopping_cost)
    #auc['ALL WHIT'] += compute_auc(L_whiten['OMP'][0], L_whiten['OMP'][1])

  for _, key in enumerate(auc):
    auc[key] /= np.float64(len(vec_set_ids))
  
elif exp_id == 2:
  auc = dict([ ('w/ cost', 0), ('w/o cost', 0), ('Oracle', 0) ])
  for _, set_id in enumerate(vec_set_ids):
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

    d_model = np.load(yahoo_common.filename_model(set_id, partition_id, 
                                                  group_size, l2_lam, False))
    stopping_cost = yahoo_common.compute_stopping_cost(alpha, d_model)

    oracle_costs, oracle_losses = compute_oracle(L_cost['FR'][0], L_cost['FR'][1])
    auc['Oracle'] += compute_auc(oracle_costs, oracle_losses, stopping_cost)
    auc['w/ cost'] += compute_auc(L_cost['OMP'][0], L_cost['OMP'][1], stopping_cost)
    auc['w/o cost'] += compute_auc(L_no_cost['OMP'][0], L_no_cost['OMP'][1], stopping_cost)
  for _, key in enumerate(auc): 
    auc[key] /= np.float64(len(vec_set_ids))

elif exp_id == 3:
  methods = ['OMP', 'OMP SINGLE', 'FR', 'FR SINGLE', 'Oracle']
  auc = dict(zip(methods, np.zeros(len(methods))))
  auc['Lasso'] = 0
  for _, set_id in enumerate(vec_set_ids):
    filename = yahoo_common.filename_budget_vs_loss(set_id, partition_id, 
                                                    group_size, l2_lam, False)
    d = np.load(filename)
    L = d['L']
    L = L.item()

    d_model = np.load(yahoo_common.filename_model(set_id, partition_id, 
                                                  group_size, l2_lam, False))
    stopping_cost = yahoo_common.compute_stopping_cost(alpha, d_model)

    for _, method in enumerate(methods):
      if method=='Oracle':
        oracle_costs, oracle_losses = compute_oracle(L['FR'][0], L['FR'][1])
        auc['Oracle'] += compute_auc(oracle_costs, oracle_losses, stopping_cost)
      else:
        auc[method] += compute_auc(L[method][0], L[method][1], stopping_cost)
    d_lasso = np.load('yahoo_results/spams_%d_%d.npz', set_id, group_size)
    auc['Lasso'] += compute_auc(d_lasso['budget'], d_lasso['loss'])

  for _, method in enumerate(methods):
    auc[method] /= np.float64(len(vec_set_ids))
  auc['Lasso'] /= np.float64(len(vec_set_ids))

elif exp_id == 4:
  pass

print auc
