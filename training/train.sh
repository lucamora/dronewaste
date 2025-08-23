#!/bin/bash

# Define model architecture (yolov8, yolov12, faster)
arch="yolov8"
# Define a unique identifier for the run (default: train)
slug=""
# Get only the date because jobs can be started at different times
today=$(date +%Y-%m-%d)

# Set path to the storage folder for the dataset and results
export STORAGE="/path/to/storage"
# Set path to temporary dataset generated for each fold
export TMPDIR="/tmp"
# Set path to virtual environments folder
export ENVSDIR="envs"


if [ "$arch" == "yolov12" ]; then
    # Activate yolov12 specific venv
    source $ENVSDIR/yolov12/bin/activate
else
    # Activate generic venv
    source $ENVSDIR/waste/bin/activate
fi

# Check if slug is empty
if [ -z "$slug" ]; then
    slug="train"
fi

# Run_id: arch + today + slug
run_id="$arch"_"$today"_"$slug"

# Loop through site indices from 0 to 16 (included)
for SITE_INDEX in {0..16}; do
    # Fold_id: site_index + timestamp for uniqueness
    fold_id="${SITE_INDEX}_$(date +%s)"
    
    echo ""
    echo "========================================="
    echo "Running k-fold: $run_id"
    echo "Fold id: $fold_id"
    echo "Site index: $SITE_INDEX"
    echo "Started at: $(date)"
    echo "========================================="
    echo ""
    
    # Create necessary directories
    mkdir -p "$(dirname "$STORAGE/kfold_results/$run_id")"
    mkdir -p logs
    
    # Run the training for this site index
    # Log output to individual files for each site index
    python kfold_train.py \
        --arch "$arch" \
        --site_index "$SITE_INDEX" \
        --run_id "$fold_id" \
        --dataset_path "$STORAGE/dronewaste_v1.0/dronewaste_v1.0_global.json" \
        --tmp_dataset_path "$TMPDIR" \
        --results_path "$STORAGE/kfold_results/$run_id" \
        2>&1 | tee "logs/log_kfold_training_${fold_id}.out"
    
    # Check if the training was successful
    if [ $? -eq 0 ]; then
        echo "Site index $SITE_INDEX completed successfully at $(date)"
    else
        echo "ERROR: Site index $SITE_INDEX failed at $(date)"
    fi
    
    echo ""
done

deactivate

echo "Training completed at $(date)"
