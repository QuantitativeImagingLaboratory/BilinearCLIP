#!/bin/bash

# List of datasets to process (excluding CIFAR100 and SUN397)
DATASETS=("oxfordpet" "aircraft" "flowers102" "stanfordcars" "food101" "dtd" "eurosat" "caltech101")

# Training parameters
NUMBER=$1
BACKBONE=$2

echo "Starting training loop for ${#DATASETS[@]} datasets..."

for DS in "${DATASETS[@]}"
do
    echo "------------------------------------------------"
    echo "Currently Training: $DS"
    echo "------------------------------------------------"

    # Execute the python command
    python train.py -d "$DS" -n $NUMBER -b $BACKBONE

    echo "------------------------------------------------"
    echo "Running Evaluation: $DS"
    echo "------------------------------------------------"

    # Execute the python command
    python eval.py -d "$DS" -n $NUMBER -b $BACKBONE -o

    # Optional: Check if the last command failed
    if [ $? -eq 0 ]; then
        echo "Successfully finished $DS"
    else
        echo "Error encountered while training $DS. Moving to next..."
    fi
done

echo "All training tasks complete!"