#!/bin/bash

# Define parameter arrays
n_values=(500 2000)
g_values=(sigmoid sfun linear)
outcomes=(continuous binary cox)
models=(NeuralPLSI PLSI)
exposure_dist=(normal uniform t)

# Loop through all combinations
for model in "${models[@]}"; do
    for n in "${n_values[@]}"; do
        for g_fn in "${g_values[@]}"; do
            for outcome in "${outcomes[@]}"; do
                for dist in "${exposure_dist[@]}"; do
                    echo "Running simulation with n=$n, g_fn=$g_fn, outcome=$outcome, model=$model, exposure_dist=$dist"
                    python main_simulation.py --n_instances "$n" --g_fn "$g_fn" --outcome "$outcome" --models "$model" --exposure_dist "$dist"
                done
            done
        done
    done
done
