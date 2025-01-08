import torch
import torch.utils
import wandb
import importlib
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
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

    if not hasattr(cfg,"wandb_path") and hasattr(cfg,"model_path") :
            raise ValueError("Wandb path or model path must be provided")
    if hasattr(cfg,"wandb_path"):
        checkpoint_name, downloaded_config = download_model(cfg.wandb_path)
        # https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file
        with open_dict(cfg):
            cfg.model_path = checkpoint_name
        update_config(cfg, downloaded_config)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    
    with wandb.init(project=cfg.project_name, config=wandb_config):
        config = wandb.config
        config.device = device
        model, test_loader = make_test_setup(config)
        # Load the saved model weights
        checkpoint = torch.load(config.model_path,weights_only=True)
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
    total_amortized_mse = 0
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
            
            batch_loss, (batch_euclidean_mse, batch_amortized_mse), batch_results = tester.calculate_batch_metrics(
                outputs, targets, indices=indices
            )
            total_loss += batch_loss
            total_euclidean_mse += batch_euclidean_mse
            total_amortized_mse += batch_amortized_mse
            results.extend(batch_results)

    # Save results and metrics
    save_results(results, f"test_results_{config.test_name}.csv")
    
    # Log final metrics
    avg_loss = total_loss / len(test_loader)
    avg_mse = total_euclidean_mse / len(test_loader)
    avg_amortized_mse = total_amortized_mse / len(test_loader)
    wandb.log({
        "test_loss": avg_loss,
        "euclidean_mse": avg_mse,
        "amortized_mse": avg_amortized_mse
    })
    
    print(f"Testing completed:")
    print(f"Average test loss: {avg_loss:.4f}")
    print(f"Average Euclidean MSE: {avg_mse:.4f}")
    print(f"Average Amortized MSE: {avg_amortized_mse:.4f}")

def download_model(wandb_path):
    # Preprocessing wandb path
    if "https://wandb.ai/" in wandb_path:
        wandb_path = wandb_path.replace("https://wandb.ai/", "")
    if "/files" in wandb_path:
        wandb_path = wandb_path.replace("/files", "")
    if "/runs" in wandb_path:
        wandb_path = wandb_path.replace("/runs", "")
    if "/overview" in wandb_path:
        wandb_path = wandb_path.replace("/overview", "")
    api = wandb.Api()
    try:
        run = api.run(wandb_path)
    except wandb.errors.CommError as e:
        raise ValueError(f"Error accessing wandb run: {e}")
    files = run.files()
    config = run.config
    checkpoints = [file for file in files if ".pth" in file.name]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoints found in the specified path")
    # Sort the checkpoints number before .pth
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    # Download the latest checkpoint
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint.download(replace=True)
    return latest_checkpoint.name, config   

def update_config(cfg, downloaded_config):
    downloaded_config = OmegaConf.create(downloaded_config)
    test_dataset = downloaded_config.test_dataset
    test_dataset.args.path = cfg.test_dataset.args.path
    cfg.model = downloaded_config.model
    cfg.tester = downloaded_config.tester
    cfg.loss = downloaded_config.loss
    cfg.test_dataset = test_dataset

if __name__ == "__main__":
    model = model_test_pipeline()