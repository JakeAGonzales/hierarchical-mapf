import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean, stdev
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.animation as animation
import time
import random

# import functions from other scripts
from load_data_flow import load_data
#from load_data_congestion import load_data
from nn import MAPFCongestionModel
from utils import get_device, set_seed

""""
Evaluate the congestion/flow prediction models individually on ground truth values from the preprocessed data 

"""

def visualize_difference(pred, ground_truth, config, threshold=0.4):
    print(f"Pred shape: {pred.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
    if pred.dim() == 3 and pred.shape[0] == 1:
        pred = pred.squeeze(0)
    
    time_steps, nodes = pred.shape
    grid_size = int(nodes**0.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Create masked arrays where values below threshold are masked
    pred_map = pred[0].reshape(grid_size, grid_size).numpy()
    truth_map = ground_truth[0].reshape(grid_size, grid_size).numpy()
    
    # Create masks for values below threshold
    pred_mask = pred_map < threshold
    truth_mask = truth_map < threshold
    
    # Create masked arrays
    pred_masked = np.ma.array(pred_map, mask=pred_mask)
    truth_masked = np.ma.array(truth_map, mask=truth_mask)
    
    # Use a colormap with transparency
    cmap = plt.cm.Blues
    cmap.set_bad(alpha=0)  # Make masked values transparent
    
    im1 = ax1.imshow(pred_masked, cmap=cmap, vmin=0, vmax=1, animated=True)
    im2 = ax2.imshow(truth_masked, cmap=cmap, vmin=0, vmax=1, animated=True)
    
    ax1.set_title("Predicted")
    ax2.set_title("Ground Truth")
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add a suptitle for the time step
    suptitle = fig.suptitle('', fontsize=16)
    
    plt.tight_layout()
    
    def update(frame):
        # Update masked arrays for each frame
        pred_map = pred[frame].reshape(grid_size, grid_size).numpy()
        truth_map = ground_truth[frame].reshape(grid_size, grid_size).numpy()
        
        pred_mask = pred_map < threshold
        truth_mask = truth_map < threshold
        
        pred_masked = np.ma.array(pred_map, mask=pred_mask)
        truth_masked = np.ma.array(truth_map, mask=truth_mask)
        
        im1.set_array(pred_masked)
        im2.set_array(truth_masked)
        
        # Update the time step in the suptitle
        suptitle.set_text(f'Time Step: {frame + 1}')
        
        return im1, im2, suptitle
    
    anim = animation.FuncAnimation(fig, update, frames=time_steps, interval=500, blit=False)
    anim.save('single_instance_comparison.gif', writer='pillow', fps=2)
    print("Visualization saved as 'single_instance_comparison.gif'")
    plt.close()

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

def evaluate_instance(model, data, config, scaling_factor=5.0):
    device = next(model.parameters()).device
    data = data.to(device)
    
    # warm-up run
    with torch.no_grad():
        _ = model(data)
    
    # measure inference time
    start_time = time.perf_counter()
    with torch.no_grad():
        pred = model(data)
    end_time = time.perf_counter()
    
    inference_time = (end_time - start_time) * 1000  # milliseconds
    
    # Apply scaling factor to the predictions
    scaled_pred = pred * scaling_factor
    
    print(f"Shape of data.y: {data.y.shape}")
    print(f"Shape of pred: {scaled_pred.shape}")
    
    mse = torch.mean((data.y - scaled_pred.squeeze(0))**2).item()
    mae = torch.mean(torch.abs(data.y - scaled_pred.squeeze(0))).item()
    
    return mse, mae, scaled_pred.cpu(), data.y.cpu(), inference_time


def evaluate_model(model, test_loader, config):
    device = get_device()
    model.to(device)
    model.eval()

    mse_list, mae_list, inference_time_list = [], [], []
    all_preds, all_labels = [], []

    for batch in tqdm(test_loader, desc="Evaluating"):
        mse, mae, pred, label, inference_time = evaluate_instance(model, batch, config)
        mse_list.append(mse)
        mae_list.append(mae)
        inference_time_list.append(inference_time)
        all_preds.append(pred)
        all_labels.append(label)

    print(f"Average MSE: {mean(mse_list):.4f} (±{stdev(mse_list):.4f})")
    print(f"Average MAE: {mean(mae_list):.4f} (±{stdev(mae_list):.4f})")
    print(f"Average Inference Time: {mean(inference_time_list):.2f}ms (±{stdev(inference_time_list):.2f}ms)")

    return all_preds, all_labels

def create_gif(all_preds, all_labels, config):
    pred = all_preds[0].squeeze(0)
    label = all_labels[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    pred_map = pred[0].reshape(config['grid_size'], config['grid_size'])
    label_map = label[0].reshape(config['grid_size'], config['grid_size'])
    
    im1 = ax1.imshow(pred_map, cmap='Blues', vmin=0, vmax=1, animated=True)
    im2 = ax2.imshow(label_map, cmap='Blues', vmin=0, vmax=1, animated=True)
    
    ax1.set_title("Predicted")
    ax2.set_title("Ground Truth")
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    def update(frame):
        pred_map = pred[frame].reshape(config['grid_size'], config['grid_size'])
        label_map = label[frame].reshape(config['grid_size'], config['grid_size'])
        
        im1.set_array(pred_map)
        im2.set_array(label_map)
        
        return im1, im2
    
    anim = animation.FuncAnimation(fig, update, frames=config['max_time_steps'], interval=500, blit=True)
    anim.save('congestion_comparison.gif', writer='pillow', fps=2)
    print("Visualization saved as 'congestion_comparison.gif'")
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
        "num_attention_heads": 1,
        "batch_size": 16,
        "dropout_rate": 0.3,
        "max_time_steps": 63,        # make sure to change max time steps and grid size
        "data_folder": "data/32x32/test",
        "grid_size": 32,
    }

    device = get_device()
    
    test_dataset, _ = load_data(config['data_folder'], config['grid_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model_path = "models/32x32/flow_model.pth"  
    model = load_model(model_path, config, device)

    
    specific_index = random.randint(1, 800) 
    single_instance = test_dataset[specific_index]
    single_instance = Batch.from_data_list([single_instance])  

    print(f"Evaluating instance {specific_index} out of {len(test_dataset)}")

    mse, mae, pred, ground_truth, inference_time = evaluate_instance(model, single_instance, config)

    print(f"MSE for single instance: {mse:.4f}")
    print(f"MAE for single instance: {mae:.4f}")
    print(f"Inference time for single instance: {inference_time:.2f} ms")

    visualize_difference(pred, ground_truth, config, threshold=0.0) 

    print("Evaluation complete. Check 'single_instance_comparison.gif' for visualization.")
