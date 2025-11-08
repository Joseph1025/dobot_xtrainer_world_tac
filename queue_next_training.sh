#!/bin/bash

# Script to wait for current training to finish and then start the next one

echo "Searching for running training process..."

# Find the PID of the current model_train.py process
CURRENT_PID=$(pgrep -f "model_train.py" | head -1)

if [ -z "$CURRENT_PID" ]; then
    echo "No training process found running."
    echo "Starting train_actjepa_baseline_vitl.sh immediately..."
    exec ./train_actjepa_baseline_vitl.sh
else
    echo "Found training process with PID: $CURRENT_PID"
    echo "Waiting for it to complete..."
    
    # Wait for the process to finish
    while kill -0 $CURRENT_PID 2>/dev/null; do
        sleep 600  # Check every 10 minutes
        echo "$(date): Still waiting for PID $CURRENT_PID to finish..."
    done
    
    echo "$(date): Previous training completed!"
    echo "Starting train_actjepa_baseline_vitl.sh now..."
    
    # Start the new training
    exec ./train_actjepa_baseline_vitl.sh
fi

