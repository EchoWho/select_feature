#!/bin/bash

dataset=yahoo
log=${dataset}_lambda_`date +%m_%d_%y`.log
echo "--------------------------------" >> $log
echo "Lambda experiment on ${dataset} `date`" >> $log
for g in 5 10 15 20; do
  for lam in 1e-6 1e-5 1e-4 1e-3 1e-2 5e-2; do
# Training 
    #cmd_str="python ${dataset}.py 2 $g $lam"
    #echo $cmd_str
    #echo $cmd_str >> $log
    #eval $cmd_str >> $log

# Testing
    cmd_str="python ${dataset}_classify.py yahoo_data/set2.valid.txt 2 $g $lam"
    echo $cmd_str
    echo $cmd_str >> $log
    eval $cmd_str >> $log
  done
done
