import torch
import torch.utils
import wandb
from tqdm import tqdm
import importlib
import pandas as pd
import torch.amp as amp
import hydra
from omegaconf import DictConfig, OmegaConf
import os

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

def xavier_init(model):
    # Xavier initialization for weights
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

@hydra.main(config_path="../configs", config_name="config",version_base=None)
def model_pipeline(cfg: DictConfig):
    downloaded_config = None
    start_epoch = 0
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    if hasattr(cfg, 'wandb_path'):
        checkpoint_path, start_epoch, downloaded_config = load_latest_checkpoint(cfg.wandb_path)
        downloaded_config = OmegaConf.create(downloaded_config)
        wandb_config = OmegaConf.to_container(downloaded_config, resolve=True)
    # TODO: Find a better way to do this
    wandb_config["epochs"] = cfg["epochs"]
    with wandb.init(project=cfg.project_name, config=wandb_config):
        config = wandb.config
        config.device = device
        model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, scaler = make(config)
        print(model)
        if downloaded_config:
            checkpoint = torch.load(checkpoint_path,weights_only=True)
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("No checkpoint found. Starting training from scratch.")
                xavier_init(model)
        else:
            print("No existing wandb path. Starting from scratch")
            xavier_init(model)
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, config, start_epoch)
        test(model, test_loader, config)

    return model

def make(config):
    # Make the data
    train, val = get_data(config.train_dataset)
    test = get_data(config.test_dataset,train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    val_loader = make_loader(val, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = instantiate_class(config.model)
    model.to(device)
        # Make the loss and optimizer
    criterion = instantiate_class(config.loss)
    optimizer = torch.optim.Adam(
    model.parameters(), lr=config.learning_rate,eps=1e-8,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = amp.GradScaler()
    return model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, scaler

def get_data(dataset_config, train=True):
    dataset = instantiate_class(dataset_config)
    if not train:
        return dataset
    train_data,val_data = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)), int(0.1 * len(dataset))])
    return train_data, val_data

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         num_workers=4)
    return loader
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, config, start_epoch):
    wandb.watch(model, criterion, log="all", log_freq=10)
    example_counter = 0  # number of examples seen
    batch_counter = 0
    print("Epochs in this run: ", config.epochs)
    for epoch in tqdm(range(start_epoch,start_epoch + config.epochs)):
        model.train()
        for data, targets in train_loader:
            
            loss = train_batch(data, targets, model, optimizer, criterion, scaler)
            example_counter +=  len(data)
            batch_counter += 1

            # Report metrics every 25th batch
            if ((batch_counter + 1) % 25) == 0:
                train_log(loss, example_counter, epoch)

        # Validate the model        
        val_loss, euclidean_mse, amortized_mse = validate(model, val_loader, config)
        scheduler.step()
        wandb.log({"epoch": epoch, "val_loss": val_loss, "euclidean_mse": euclidean_mse, "amortized_mse": amortized_mse})
        if (epoch + 1) % config.checkpoint_frequency == 0:
           save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss)
    save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss)
def validate(model, val_loader, config):
    model.eval()
    tester = instantiate_class(config.tester)
    tester.set_criterion(instantiate_class(config.loss))
    val_loss = 0.0
    euclidean_loss = 0.0
    amortized_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            batch_loss, (euclidean_mse, amortized_mse), _ = tester.calculate_batch_metrics(outputs, targets)
            euclidean_loss += euclidean_mse
            amortized_loss += amortized_mse
            val_loss += batch_loss.item()
    return val_loss / len(val_loader), euclidean_loss / len(val_loader), amortized_loss / len(val_loader)


def train_batch(data, targets, model, optimizer, criterion, scaler):
    data, targets = data.to(device), targets.to(device)
    # Forward pass ➡
    with amp.autocast("cuda"):
        outputs = model(data)
        loss = criterion(outputs, targets)
    # Backward pass ⬅
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss

def train_log(loss, example_counter, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_counter)
    print(f"Loss after {str(example_counter).zfill(5)} examples: {loss:.3f}")

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    wandb.save(filename)

def test(model, test_loader, config):
    model.eval()
    tester = instantiate_class(config.tester)
    tester.set_criterion(instantiate_class(config.loss))
    total_loss = 0
    total_euclidean_mse = 0
    total_amortized_mse = 0
    results = []
    example_input = None
    # Run the model on some test examples
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            if example_input is None:
                example_input = data
            batch_loss, (batch_euclidean_mse, batch_amortized_mse), batch_results = tester.calculate_batch_metrics(outputs, targets)
            total_loss += batch_loss
            total_euclidean_mse += batch_euclidean_mse
            total_amortized_mse += batch_amortized_mse
            results.extend(batch_results)
    save_results(results, "test_results.csv")
    # Save the model in ONNX format because it's cool
    torch.onnx.export(model, example_input, "model.onnx")
    wandb.save("model.onnx")
    wandb.log({"test_loss": total_loss / len(test_loader), "euclidean_mse": total_euclidean_mse / len(test_loader), "amortized_mse": total_amortized_mse / len(test_loader)})

def load_latest_checkpoint(wandb_path):
    if "https://wandb.ai/" in wandb_path:
        wandb_path = wandb_path.replace("https://wandb.ai/", "")
    if "/files" in wandb_path:
        wandb_path = wandb_path.replace("/files", "")
    if "/runs" in wandb_path:
        wandb_path = wandb_path.replace("/runs", "")
    if "/overview" in wandb_path:
        wandb_path = wandb_path.replace("/overview", "")
    # I think this does not work for now :)
    api = wandb.Api()
    run = api.run(wandb_path)
    files = run.files()
    checkpoints = [f for f in files if f.name.startswith("ckpt_epoch_")]
    if not checkpoints:
        return None, 0, None
        # Sort the checkpoints number before .pth
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    # Download the latest checkpoint
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint.download(replace=True)
    epoch = int(latest_checkpoint.name.split("_")[-1].split(".")[0])
    print("Last checkpoint found at epoch", epoch)
    # Get config from the run
    config = run.config
    return latest_checkpoint.name, epoch, config

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, f"ckpt_epoch_{epoch+1}.pth")
    wandb.save(f"ckpt_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    # I am calling this because torch warns me about it
    torch.multiprocessing.set_start_method('spawn')
    model = model_pipeline()