#!/bin/bash

# experiment settings
DATASET=StanfordCars
N_CLASS=196

# hard coded inputs
GPUID='4'
CONFIG=configs/stanfordcars_prompt.yaml
REPEAT=1
OVERWRITE=0

# hyperparameter arrays
LR=0.005
SCHEDULE=50
EMA_COEFF=0.7
SEED_LIST=(1 2 3)

# Set delay between experiments (in seconds)
DELAY_BETWEEN_EXPERIMENTS=10  # Adjust this value as needed

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

for seed in "${SEED_LIST[@]}"
    do
        # save directory
        OUTDIR="./checkpoints/${DATASET}/seed${seed}"
        mkdir -p $OUTDIR

        # Create unique log file name
        LOG_FILE="${LOG_DIR}/${DATASET}/seed${seed}.log"

        echo "Starting experiment with seed=$seed"
        
        nohup python -u run.py \
            --config $CONFIG \
            --gpuid $GPUID \
            --repeat $REPEAT \
            --overwrite $OVERWRITE \
            --learner_type prompt \
            --learner_name APT_Learner \
            --prompt_param 0.01 \
            --lr $LR \
            --seed $seed \
            --ema_coeff $EMA_COEFF \
            --schedule $SCHEDULE \
            --log_dir ${OUTDIR} > "$LOG_FILE" 2>&1 &

        # Store the PID of the background process
        PID=$!
        
        # Wait for process to complete
        wait $PID
        
        # Check if process completed successfully
        if [ $? -eq 0 ]; then
            echo "Experiment completed successfully"
        else
            echo "Experiment failed"
        fi

        rm -rf ${OUTDIR}/models
        
        echo "----------------------------------------"
        
        # Add delay before next experiment
        if [ $current -lt $total_experiments ]; then
            echo "Waiting for $DELAY_BETWEEN_EXPERIMENTS seconds before next experiment..."
            sleep $DELAY_BETWEEN_EXPERIMENTS
        fi
    done

echo "All experiments completed!"
exit 0