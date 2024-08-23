#!/bin/bash
python /home/lib/baco/baco/plot/plot_optimization_results.py -j /home/lib/Performance_Exploration/search_settings.json -i /home/results -l "batch matmul" --y_label "runtime (s)" --title "" -o "/home/results/runtime_evolution.pdf" || { echo "\nError: Failed to execute the plotting script. Did you run performance_exploration.py?"; exit 1; }
echo "Runtime evolution plot generated successfully, check /home/results"
