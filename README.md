# Multi-Tx-Modular-Trainer

## Description
This project is a template for training multi-tx models using a modular approach.

## Installation
To install this project on your cluster, follow these steps:

```bash
# Clone the repository
git clone https://github.com/efemantarci/modular-train.git
# Navigate to the project directory
cd modular-train
```
This project uses Weights and Biases for logging the data. Add your W&B API key in the [config.env](configs/config.env) file. Then, write your username in the necessary parts of the [train.sh](train.sh) and [test.sh](test.sh) files.
If you want to change the docker container, you must use a container with wandb and hydra installed.

## Usage
The project consists of two main scripts:

- **[train.sh](train.sh)**: This script is used to train the model. You can run it with the default configuration or specify a different configuration file.
- **[test.sh](test.sh)**: This script is used to test the model after training.

Example usage:

```bash
# Train the model
sbatch train.sh
# Train the model with a specific configuration
sbatch train.sh --config_name=config_raw3d_prediction epochs=100
# Test the model
sbatch test.sh
```

## Configurations
The models are trained in a modular manner. If you want to train a different model, you need to add your functions to the following files, in order of importance:

- **[models.py](models.py)**: Add your neural network class that extends `nn.Module` here.
- **[datasets.py](datasets.py)**: Specify how data from the dataset is pulled.
- **[testers.py](testers.py)**: Define how you save test results here.
- **[losses.py](losses.py)**: Write your loss function here.

Then, specify these functions as arguments in the configuration file. Two example configurations are provided:

- `config.yaml`: The default configuration, which trains the model based on the distance between two 3D points.
- `config_raw3d_prediction.yaml`: Trains the model by predicting six floats, directly representing points in space.

You should check these configs and look at the functions defined on them and make slight changes. You probably only need to change model at models.py
