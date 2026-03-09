#!/bin/bash

# --- Check Arguments ---
if [ "$#" -ne 1 ]; then
    echo "Usage: ./reproduce_few_shot_results.sh <backbone>"
    echo "Example: ./reproduce_few_shot_results.sh vit16"
    exit 1
fi

BACKBONE=$1
SHOT_COUNTS=(1 2 4 8 16)
RESULTS_DIR="few_shot_results_$(date +%Y%m%d)"

mkdir -p "$RESULTS_DIR"

echo "===================================================="
echo " BiCLIP Full Few-Shot Sweep"
echo " Backbone: $BACKBONE"
echo " Results Directory: $RESULTS_DIR"
echo "===================================================="

for SHOTS in "${SHOT_COUNTS[@]}"
do
    echo -e "\n>>> Starting $SHOTS-shot experiments..."

    # Run the main reproduction script for the current shot count
    # We pass the backbone and shots to the existing script
    ./reproduce_main_results.sh "$SHOTS" "$BACKBONE"

    # Move the resulting log file to our results directory for organization
    # Assuming the main script creates a file named reproduction_*.log
    LATEST_LOG=$(ls -t reproduction_*.log | head -1)
    mv "$LATEST_LOG" "$RESULTS_DIR/${SHOTS}_shot_results.log"

    echo ">>> Finished $SHOTS-shot. Log saved to $RESULTS_DIR/${SHOTS}_shot_results.log"
done

echo -e "\n===================================================="
echo " Sweep Complete! All logs are in $RESULTS_DIR"
echo " You can now run: python visualization.py --few-shot --dir $RESULTS_DIR"
echo "===================================================="