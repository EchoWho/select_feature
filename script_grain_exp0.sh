#!/bin/bash

dataset=grain
log=${dataset}_lambda_`date +%m_%d_%y`.log
echo "--------------------------------" >> $log
echo "Lambda experiment on ${dataset} `date`" >> $log
set_id=2
lam=1e-7
for lam in 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 5e-2; do
# Training 
    cmd_str="python ${dataset}.py $set_id $lam"
    echo $cmd_str
    echo $cmd_str >> $log
    eval $cmd_str >> $log

# Testing
#    cmd_str="python ${dataset}_classify.py 2 2 $lam"
#    echo $cmd_str
#    echo $cmd_str >> $log
#    eval $cmd_str >> $log
done
