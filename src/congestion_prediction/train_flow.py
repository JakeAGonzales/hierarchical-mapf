import os
import time
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from load_data_flow import load_data
from nn import MAPFCongestionModel
from utils import get_device, set_seed

"""
Training script for the flow prediction model.  
"""

CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'models', 'model_checkpoints')
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

def get_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    pyg_dataset, max_time_steps = load_data(config['data_folder'], config['grid_size'])
    config['max_time_steps'] = max_time_steps
    train_size = int(0.8 * len(pyg_dataset))
    train_dataset, val_dataset = pyg_dataset[:train_size], pyg_dataset[train_size:]
    
    def collate_fn(batch: List[Data]) -> Data:
        batch_x = torch.cat([data.x for data in batch], dim=0)
        batch_y = torch.cat([data.y for data in batch], dim=0)
        
        batch_size = len(batch)
        time_steps, num_nodes = batch[0].y.shape
        
        edge_indices = [data.edge_index + i * num_nodes for i, data in enumerate(batch)]
        batch_edge_index = torch.cat(edge_indices, dim=1)
        
        return Data(x=batch_x, edge_index=batch_edge_index, y=batch_y, num_nodes=num_nodes*batch_size)
    
    return (
        DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn),
        DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    )

def train_model(config: Dict) -> Tuple[List[float], List[float]]:
    device = get_device()
    print("using device: ", device)
    model = MAPFCongestionModel(
        input_dim=config['in_channels'],
        hidden_dim=config['hidden_channels'],
        output_dim=config['out_channels'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        max_time_steps=config['max_time_steps'],
        dropout_rate=config['dropout_rate']  
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loader, val_loader = get_data_loaders(config)
    best_val_mse = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(config['num_of_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{config["num_of_epochs"]} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}')

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f'Learning rate changed from {old_lr} to {new_lr}')
        
        scheduler.step(val_loss)
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            save_model(model, optimizer, epoch, val_loss, config)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config)

    return train_losses, val_losses

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                loss_fn: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.view(-1, out.size(-1)), batch.y.view(-1, batch.y.size(-1)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

def validate_epoch(model: nn.Module, val_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out.view(-1, out.size(-1)), batch.y.view(-1, batch.y.size(-1)))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(val_loader.dataset)

def save_model(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_loss: float, config: Dict) -> None:
    model_name = f"mapf_flow_model_hidden{config['hidden_channels']}_lr{config['lr']}_epoch{epoch+1}_valMSE{val_loss:.4f}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, os.path.join(CHECKPOINTS_PATH, model_name))
    print(f"Saved best model: {model_name}")

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                    train_loss: float, val_loss: float, config: Dict) -> None:
    checkpoint_name = f"checkpoint_epoch{epoch+1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, os.path.join(CHECKPOINTS_PATH, checkpoint_name))
    print(f"Saved checkpoint: {checkpoint_name}")

def plot_training_progress(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train MSE')
    plt.plot(epochs, val_losses, label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and Validation MSE for Flow Prediction')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    set_seed(69)

    config = {
        "num_of_epochs": 25,
        "lr": 0.00001,
        "weight_decay": 5e-4,
        "hidden_channels": 8,
        "in_channels": 3,
        "out_channels": 1,
        "num_gnn_layers": 1,
        "num_attention_heads": 2,
        "batch_size": 32,
        "dropout_rate": 0.2,
        "max_time_steps": 16,           # make sure to change max time steps and grid size
        "data_folder": "data/8x8/train",
        "grid_size": 8,
    }
    
    train_losses, val_losses = train_model(config)
    plot_training_progress(train_losses, val_losses)