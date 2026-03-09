#!/bin/bash

# --- Check Arguments ---
if [ "$#" -ne 2 ]; then
    echo "Usage: ./reproduce_main_results.sh <num_shots> <backbone>"
    echo "Example: ./reproduce_main_results.sh 16 vit16"
    exit 1
fi

SHOTS=$1
BACKBONE=$2
LOG_FILE="reproduction_$(date +%Y%m%d_%H%M%S).log"

# List of datasets from the BiCLIP paper
DATASETS=("oxfordpet" "aircraft" "flowers102" "stanfordcars" "food101" "dtd" "eurosat" "imagenet" "sun397" "ucf101" "caltech101")

echo "===================================================="
echo " BiCLIP & BiSigLIP Reproduction Script"
echo " Configuration: ${SHOTS}-shot | Backbone: ${BACKBONE}"
echo " Logging to: ${LOG_FILE}"
echo "===================================================="

# --- 1. CLIP + BiCLIP Section ---
echo -e "\n[Phase 1/2] Training and Evaluating CLIP-based Models..."
for DS in "${DATASETS[@]}"
do
    echo "Processing $DS..."

    # Train BiCLIP
    python train.py -d "$DS" -n "$SHOTS" -b "$BACKBONE" >> "$LOG_FILE" 2>&1

    # Evaluate and append results
    python eval.py -d "$DS" -n "$SHOTS" -b "$BACKBONE" -o >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Successfully finished CLIP adaptation for $DS"
    else
        echo "Error: CLIP adaptation failed for $DS. Check $LOG_FILE for details."
    fi
done

# --- 2. SigLIP + BiSigLIP Section ---
echo -e "\n[Phase 2/2] Training and Evaluating SigLIP-based Models..."
for DS in "${DATASETS[@]}"
do
    echo "Processing $DS..."

    # Train BiSigLIP
    python train_siglip.py -d "$DS" -n "$SHOTS" -b "$BACKBONE" >> "$LOG_FILE" 2>&1

    # Evaluate and append results
    python eval_siglip.py -d "$DS" -n "$SHOTS" -b "$BACKBONE" -o >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "Successfully finished SigLIP adaptation for $DS"
    else
        echo "Error: SigLIP adaptation failed for $DS. Check $LOG_FILE for details."
    fi
done

echo -e "\n===================================================="
echo " All tasks complete! Final results are logged in: ${LOG_FILE}"
echo "===================================================="