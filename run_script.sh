#!/bin/bash

# Training agent with varying random seed on the same Unity environment seed.
# Doing this in a shell script since cannot reset Unity environment seed 
# without restarting kernel in Jupyter notebook. 

# List of random seeds and configurations for agent
agent_seeds='0 1 2 3 4 5 6 7'
configs='0 1 2 3 4'

# Loop through configs
for config in $configs
do
    # Loop through seeds and train
    for seed in $agent_seeds
    do
        python train.py $seed $config
    done
done
