#!/bin/bash

set -e

# Args
SKIP_DOWNLOAD=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-download) SKIP_DOWNLOAD=true ;;
        *) echo -e "${RED}Unknown parameter passed: $1${NC}" ; exit 1 ;;
    esac
    shift
done

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

MODELDIR="/home/models"
MODELEXTENSION="_tosa.mlir"

check_last_line_for_error() {
    local log_file="$1"
    local search_string="$2"

    # Get the last line of the log file
    last_line=$(tail -n 1 "$log_file")

    # Check if the last line contains the specific string using grep for pattern matching
    if echo "$last_line" | grep -qF "$search_string"; then
        return 0
    else
        return 1
    fi
}

echo "######################## CASE STUDY 1 #######################"
echo "################## Compile time comparison ##################"
if [ "$SKIP_DOWNLOAD" = true ]; then
echo "Skipping downloading models for case study 1"
else
echo "Downloading models for case study 1"
bash /home/scripts/show_spinner.sh "python /home/scripts/get_models.py --all"
fi

echo -e "Benching compile time for all models in /home/models with the suffix _tosa.mlir"

cs1_output_file="/home/results/CS1_compile_time_comparison.txt"

if [ -d "$MODELDIR" ] && [ "$(ls -A $MODELDIR)" ]; then
    # Loop through each model in the directory
    for file in "$MODELDIR"/*; do
        if [[ "$file" == *"$MODELEXTENSION" ]]; then
            echo "Benchmarking converted model: $file"
            echo "------------- $file -------------" 
            output=$(python /home/scripts/bench_compile_time.py $file --pass-pipeline="builtin.module(func.func(tosa-optional-decompositions), canonicalize, func.func(tosa-infer-shapes, tosa-make-broadcastable, tosa-to-linalg-named), canonicalize, func.func(tosa-layerwise-constant-fold, tosa-make-broadcastable), tosa-validate, func.func(tosa-to-linalg, tosa-to-arith, tosa-to-tensor), linalg-fuse-elementwise-ops, one-shot-bufferize)")
            echo -e "$output"
            echo -e "-------- Benchmarking $file --------" >> $cs1_output_file
            echo -e "$output" >> $cs1_output_file
            echo -e "-------------------------------------------------------------" >> $cs1_output_file
            echo "-------------------------------------------------------------" 
        else
            echo "Skipping file: $file"
        fi
    done
else
    echo "No models with suffix $MODELEXTENSION found!"
fi

echo -e "\n####################### CASE STUDY 3 ########################"
echo "#### Identifying problematic patterns in Enzyme on llama ####"
cs3_output_file="/home/results/CS3_llama_autodiff_comparison.txt"
VENV_PATH="/home/.venv39"
CS3COMMAND="python /home/lib/Enzyme-JaX/test/llama.py"

# This needs a different Python version (3.9) from the rest
# Activate the virtual environment and run the command in a subshell
(
    source "$VENV_PATH/bin/activate"
    $CS3COMMAND
    deactivate
) > $cs3_output_file 2>&1
echo -e "$cs3_output_file"


echo -e "\n####################### CASE STUDY 4 ########################"
echo "#### Comparing Resnet 50 layer with OpenMP and transform ####"
cs4_output_file="/home/results/CS4_batch_matmul_omp_vs_transform_comparison.txt"

/home/bin/batch_matmul > $cs4_output_file
echo -e "$cs4_output_file"

echo -e "\n####################### CASE STUDY 5 ########################"
echo "################## Integration of autotuning ##################"
echo -e "Starting search with the following configuration.\nThis might take a while."
echo -e "Modify the following file to change the search settings.\n In particular the value of optimization_iterations."
echo "> head -n 9 /home/lib/Performance_Exploration/search_settings.json"
head -n 9 /home/lib/Performance_Exploration/search_settings.json

search_command="python /home/scripts/performance_exploration.py > /home/results/logs/search_log.txt 2>&1 3>/home/results/logs/search_errors_log.txt"
if bash /home/scripts/show_spinner.sh "$search_command"; then
    echo "Search process completed successfully."
    echo "Plotting the optimization results, check /home/results for the pdf file."
    python /home/lib/baco/baco/plot/plot_optimization_results.py -j /home/lib/Performance_Exploration/search_settings.json -i /home/results -l "batch matmul" --y_label "runtime (s)" --title "" -o "/home/results/CS5_runtime_evolution.pdf" > /dev/null 2>&1
else
    if check_last_line_for_error "/home/results/logs/search_log.txt" "Input data contains NaN values"; then
        echo -e "${RED}Error${NC} during the search process: \nBaco could not get enough samples to construct the search space.\nPlease try increasing number_of_samples in /home/lib/Performance_Exploration/search_settings.json"
    else
        echo "${RED}Unknown Error${NC} during the search process, check the log files in /home/results/logs for more details."
    fi
fi
