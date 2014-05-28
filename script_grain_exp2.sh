#!/bin/bash

#### Require ignore_cost=True, whiten=False

dataset=grain
log=${dataset}_lambda_`date +%m_%d_%y`.log
echo "--------------------------------" >> $log
echo "Lambda experiment on ${dataset} `date`" >> $log
lam=1e-7
for set_id in 1 2 3 4 5; do
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
