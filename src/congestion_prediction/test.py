# test.py
import torch
import numpy as np
from statistics import mean, stdev
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import time
import random

from load_data_congestion import load_data
from nn import MAPFCongestionModel
from utils import get_device, set_seed

# get average results on large amount of test problems.

def load_model(model_path, config, device):
    model = MAPFCongestionModel(
        input_dim=config['in_channels'],
        hidden_dim=config['hidden_channels'],
        output_dim=config['out_channels'],
        num_gnn_layers=config['num_gnn_layers'],
        num_attention_heads=config['num_attention_heads'],
        max_time_steps=config['max_time_steps'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def evaluate_instance(model, data, config):
    device = next(model.parameters()).device
    data = data.to(device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        pred = model(data)
    end_time = time.perf_counter()
    
    inference_time = (end_time - start_time) * 1000
    
    mse = torch.mean((data.y - pred.squeeze(0))**2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(data.y - pred.squeeze(0))).item()
    
    return mse, rmse, mae, inference_time

def batch_evaluate_model(config, num_evaluations=1000):
    device = get_device()
    print(f"Using device: {device}")
    
    test_dataset, _ = load_data(config['data_folder'], config['grid_size'])
    model = load_model(config['model_path'], config, device)
    
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'inference_time': []
    }
    
    dataset_size = len(test_dataset)
    for i in tqdm(range(num_evaluations), desc="Running evaluations"):
        idx = random.randint(0, dataset_size - 1)
        instance = test_dataset[idx]
        instance = Batch.from_data_list([instance])
        
        mse, rmse, mae, inference_time = evaluate_instance(model, instance, config)
        
        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['inference_time'].append(inference_time)
    
    stats = {}
    for metric, values in metrics.items():
        stats[metric] = {
            'mean': mean(values),
            'std': stdev(values),
            'min': min(values),
            'max': max(values)
        }
    
    return stats

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
        "num_attention_heads": 1,
        "batch_size": 16,
        "dropout_rate": 0.3,
        "max_time_steps": 63,        # make sure to change max time steps and grid size
        "data_folder": "data/32x32/test",
        "grid_size": 32,
        "model_path": "models/32x32/flow_model.pth"
    }
     
    
    stats = batch_evaluate_model(config, num_evaluations=1000)
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, values in stats.items():
        print(f"\n{metric.upper()}:")
        print(f"Mean: {values['mean']:.4f}")
        print(f"Std:  {values['std']:.4f}")
        print(f"Min:  {values['min']:.4f}")
        print(f"Max:  {values['max']:.4f}")
