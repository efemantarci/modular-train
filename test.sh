#!/bin/bash
#SBATCH --container-image ghcr.io\#efe-docker-container/default:latest
# Mounting the datasets directory to the container. Yeah it looks cursed, but it works.
#SBATCH --container-mounts /home/{your-username-here}:/mnt/{your-username-here},/storage/datasets:/storage/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --job-name=ModelTest

cd /mnt/{your-username-here}/modular-train
# Source the configuration file
source configs/config.env
export WANDB_API_KEY=$WANDB_API_KEY
wandb login

python src/modular_tester_regression.py
