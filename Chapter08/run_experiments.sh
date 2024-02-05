#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/..

if ! command -v nvidia-smi &> /dev/null
then
    echo "CUDA does not seem to be available on this system. Exiting."
    exit 1
fi

scripts=(
    "01_dqn_basic.py"
    "02_dqn_n_steps.py"
    "03_dqn_double.py"
    "04_dqn_noisy_net.py"
    "05_dqn_prio_replay.py"
    "06_dqn_dueling.py"
    "07_dqn_distrib.py"
    "08_dqn_rainbow.py"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script with CUDA..."
        timeout 7200 python "$script" --cuda
        echo "$script finished."
        echo "--------------------------"
    else
        echo "Script $script does not exist."
    fi
done

echo "All specified experiments have been run."
