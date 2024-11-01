import torch
import torch.utils
import wandb
import importlib
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_class(dict_config):
    # Loading a class from a string
    module_string = dict_config["class"]
    args = dict_config.get("args", {})
    module_path, class_name = module_string.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    if args:
        return class_(**args)
    return class_()

@hydra.main(config_path="../configs", config_name="test_config", version_base=None)
def model_test_pipeline(cfg: DictConfig):
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(project=cfg.project_name, config=wandb_config):
        config = wandb.config
        config.device = device
        
        # Load model and test data
        model, test_loader = make_test_setup(config)
        
        # Load the saved model weights
        if not hasattr(config, 'model_path'):
            raise ValueError("Model path must be specified in config for testing")
            
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
        test_model(model, test_loader, config)

    return model

def make_test_setup(config):
    # Make the test data
    test_dataset = get_test_data(config.test_dataset)
    test_loader = make_loader(test_dataset, batch_size=config.batch_size)

    model = instantiate_class(config.model)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, test_loader

def get_test_data(dataset_config):
    return instantiate_class(dataset_config)

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for testing
        num_workers=4
    )
    return loader

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    wandb.save(filename)

def test_model(model, test_loader, config):
    tester = instantiate_class(config.tester)
    tester.set_criterion(instantiate_class(config.loss))
    total_loss = 0
    total_euclidean_mse = 0
    results = []
    example_input = None

    with torch.no_grad():
        for batch_idx, (data, targets) in tqdm(enumerate(test_loader)):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            if example_input is None:
                example_input = data
            
            # Calculate the actual dataset indices for this batch
            batch_size = data.size(0)
            start_idx = batch_idx * test_loader.batch_size
            indices = range(start_idx, start_idx + batch_size)
            
            batch_loss, batch_euclidean_mse, batch_results = tester.calculate_batch_metrics(
                outputs, targets, indices=indices
            )
            total_loss += batch_loss
            total_euclidean_mse += batch_euclidean_mse
            results.extend(batch_results)

    # Save results and metrics
    save_results(results, f"test_results_{config.test_name}.csv")
    
    # Log final metrics
    avg_loss = total_loss / len(test_loader)
    avg_mse = total_euclidean_mse / len(test_loader)
    wandb.log({
        "test_loss": avg_loss,
        "euclidean_mse": avg_mse
    })
    
    print(f"Testing completed:")
    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Average Euclidean MSE: {avg_mse:.4f}")

if __name__ == "__main__":
    model = model_test_pipeline()