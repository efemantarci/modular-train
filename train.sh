#!/bin/bash
#SBATCH --container-image ghcr.io\#efe-docker-container/default:latest
# Mounting the datasets directory to the container. Yeah it looks cursed, but it works.
#SBATCH --container-mounts /home/efe-mantaroglu:/mnt/efe-mantaroglu,/storage/datasets:/storage/datasets:/storage/datasets
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --job-name=CNN+LSTM

# Default values for the parameters
epochs=32
batch_size=256
learning_rate=0.001
config_name=config.yaml

# Parse command-line arguments to override defaults
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs=*)
      epochs="${1#*=}"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
      ;;
    --learning_rate=*)
      learning_rate="${1#*=}"
      ;;
    --num_of_points=*)
      num_of_points="${1#*=}"
      ;;
    --config_name=*)
      config_name="${1#*=}"
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
  shift
done

cd /mnt/efe-mantaroglu/modular_train
# Source the configuration file
source configs/config.env
export WANDB_API_KEY=$WANDB_API_KEY
wandb login

python src/modular_trainer_regression.py \
  --config-name=${config_name}\
  hydra.run.dir=. \
  epochs=${epochs} \
  batch_size=${batch_size} \
  learning_rate=${learning_rate} \