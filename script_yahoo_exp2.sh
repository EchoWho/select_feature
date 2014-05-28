#!/bin/bash

# Require ignore_cost=True whiten=False

dataset=yahoo
log=${dataset}_lambda_`date +%m_%d_%y`.log
echo "--------------------------------" >> $log
echo "Lambda experiment on ${dataset} `date`" >> $log
lam=1e-5
for g in 5 10 15 20; do
  for set_id in 1 2; do
# Training 
    cmd_str="python ${dataset}.py $set_id $g $lam"
    echo $cmd_str
    echo $cmd_str >> $log
    eval $cmd_str >> $log

# Testing
    #cmd_str="python ${dataset}_classify.py yahoo_data/set2.valid.txt 2 $g $lam"
    #echo $cmd_str
    #echo $cmd_str >> $log
    #eval $cmd_str >> $log
  done
done
