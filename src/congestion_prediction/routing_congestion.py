import os
import numpy as np
import torch
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import time
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import nnls

from multiprocessing import Pool
from functools import partial

from nn import MAPFCongestionModel as FlowModel
from nn import MAPFCongestionModel as CongestionModel
from utils import get_device, set_seed
from cbs import conflict_based_search, MAPFProblem, Environment

class SimplifiedSimulation:
    def __init__(self, map_file, time_steps, num_od_pairs):
        self.grid, self.grid_size = self.load_map(map_file)
        self.time_steps = time_steps
        self.num_od_pairs = num_od_pairs
        self.device = get_device()
        set_seed(42)
        self.flow_model = self.load_flow_model()
        self.congestion_model = self.load_congestion_model()
        self.graph = self.create_graph()
        self.timings = {}
    
    def load_map(self, map_file):
        with open(map_file, 'r') as f:
            lines = f.readlines()
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        grid = np.zeros((height, width), dtype=int)
        for i, line in enumerate(lines[4:]):
            for j, char in enumerate(line.strip()):
                if char == '.':
                    grid[i, j] = 1  
        return grid, height 
        
    def load_flow_model(self):
        config = {
            "input_dim": 3,
            "hidden_dim": 8,
            "output_dim": 1,
            "num_gnn_layers": 1,
            "num_attention_heads": 1,
            "max_time_steps": 31,
            "dropout_rate": 0.4,
        }
        model = FlowModel(**config).to(self.device)
        model_path = "models/16x16/flow_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def load_congestion_model(self):
        config = {
            "input_dim": 3,
            "hidden_dim": 8,
            "output_dim": 1,
            "num_gnn_layers": 1,
            "num_attention_heads": 1,
            "max_time_steps": 31,
            "dropout_rate": 0.2,
        }
        model = CongestionModel(**config).to(self.device)
        model_path = "models/16x16/congestion_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def create_graph(self):
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        edges_to_remove = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    edges_to_remove.extend([((i, j), n) for n in G.neighbors((i, j))])
        G.remove_edges_from(edges_to_remove)
        return G
    
    def get_boundary_positions(self):
        boundary = []
        for i in range(self.grid_size):
            if self.grid[i, 0] == 1:
                boundary.append((i, 0))
            if self.grid[i, self.grid_size - 1] == 1:
                boundary.append((i, self.grid_size - 1))
        for j in range(1, self.grid_size - 1):
            if self.grid[0, j] == 1:
                boundary.append((0, j))
            if self.grid[self.grid_size - 1, j] == 1:
                boundary.append((self.grid_size - 1, j))
        return boundary
    
    def generate_od_pairs(self):
        boundary_positions = self.get_boundary_positions()
        random.shuffle(boundary_positions)
        n_agents = random.randint(5, 14)  # change based on grid size
        if len(boundary_positions) < 2 * n_agents:
            n_agents = len(boundary_positions) // 2
        agent_pos = boundary_positions[:n_agents]
        goal_pos = boundary_positions[n_agents:2*n_agents]
        od_pairs = [((0, *start), (self.time_steps - 1, *end)) for start, end in zip(agent_pos, goal_pos)]
        return od_pairs
    
    def create_flow_input_data(self, od_pairs):
        num_nodes = self.grid_size * self.grid_size
        node_features = torch.zeros((self.time_steps, num_nodes, 3))
        for source, target in od_pairs:
            start_idx = source[1] * self.grid_size + source[2]
            end_idx = target[1] * self.grid_size + target[2]
            node_features[source[0], start_idx, 0] = 1
            node_features[target[0], end_idx, 1] = 1
            path = nx.shortest_path(self.graph, (source[1], source[2]), (target[1], target[2]))
            for t, (x, y) in enumerate(path):
                if t < self.time_steps:
                    node_idx = x * self.grid_size + y
                    node_features[t, node_idx, 2] += 1
        edge_index = self.create_edge_index()
        return Data(x=node_features.to(self.device), edge_index=edge_index.to(self.device))
    
    def create_edge_index(self):
        edges = list(self.graph.edges())
        edge_index = torch.tensor([[self.grid_size * x1 + y1, self.grid_size * x2 + y2] for (x1, y1), (x2, y2) in edges], dtype=torch.long).t().contiguous()
        return edge_index
    
    def predict_flows(self, od_pairs):
        input_data = self.create_flow_input_data(od_pairs)
        inference_start = time.time()
        with torch.no_grad():
            predicted_flows = self.flow_model(input_data)
        inference_time = time.time() - inference_start
        scaling_factor = 1.2
        predicted_flows *= scaling_factor
        self.timings['flow_inference'] = inference_time
        return predicted_flows.view(self.time_steps, self.grid_size, self.grid_size).squeeze().cpu().numpy()
    
    def create_congestion_input_data(self, predicted_flows):
        num_nodes = self.grid_size * self.grid_size
        node_features = torch.zeros((self.time_steps, num_nodes, 3))
        node_features[:, :, 2] = torch.tensor(predicted_flows.reshape(self.time_steps, -1))
        edge_index = self.create_edge_index()
        return Data(x=node_features.to(self.device), edge_index=edge_index.to(self.device))
    
    def predict_congestion(self, predicted_flows):
        input_data = self.create_congestion_input_data(predicted_flows)
        inference_start = time.time()
        with torch.no_grad():
            predicted_congestion = self.congestion_model(input_data)
        inference_time = time.time() - inference_start
        self.timings['congestion_inference'] = inference_time
        congestion = predicted_congestion.view(self.time_steps, self.grid_size, self.grid_size).squeeze().cpu().numpy()
        congestion = np.maximum(congestion, 0)
        return congestion
    
    def run_simulation(self):
        od_pairs = self.generate_od_pairs()
        predicted_flows = self.predict_flows(od_pairs)
        predicted_congestion = self.predict_congestion(predicted_flows)
        return predicted_flows, predicted_congestion, od_pairs, self.timings

def compute_ground_truth_congestion(env, goals, solution):
    grid_size = env.size[0]
    max_time_steps = max(len(path) for path in solution.paths)
    congestion = np.zeros((max_time_steps, grid_size, grid_size))
    adjacents = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
    for t in range(max_time_steps):
        for path in solution.paths:
            if t < len(path):
                x, y = path[t].pos
                for dx, dy in adjacents:
                    adj_x, adj_y = x + dx, y + dy
                    if 0 <= adj_x < grid_size and 0 <= adj_y < grid_size:
                        congestion[t, adj_x, adj_y] += 1
    max_congestion = np.max(congestion)
    if max_congestion > 0:
        congestion = congestion / max_congestion
    return congestion

def get_ground_truth_congestion(map_file, n_agents, time_steps, od_pairs):
    with open(map_file, 'r') as f:
        lines = f.readlines()
    grid_size = int(lines[1].split()[1])
    obstacle_pos = []
    for i, line in enumerate(lines[4:]):
        for j, char in enumerate(line.strip()):
            if char != '.':
                obstacle_pos.append((i, j))
    agent_pos = [od_pair[0][1:] for od_pair in od_pairs]
    goal_pos = [od_pair[1][1:] for od_pair in od_pairs]
    env = Environment((grid_size, grid_size), obstacle_pos, agent_pos)
    problem = MAPFProblem(env, goal_pos)
    solution = conflict_based_search(problem)
    if solution is None:
        raise ValueError("CBS couldn't find a solution for the given problem.")
    return compute_ground_truth_congestion(env, goal_pos, solution)

def visualize_congestion_comparison(predicted_congestion, ground_truth_congestion, config):
    max_time_steps = min(predicted_congestion.shape[0], ground_truth_congestion.shape[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    def update(frame):
        ax1.clear()
        ax2.clear()
        im1 = ax1.imshow(predicted_congestion[frame], cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title(f'Predicted Congestion at Time Step {frame + 1}')
        im2 = ax2.imshow(ground_truth_congestion[frame], cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title(f'Ground Truth Congestion at Time Step {frame + 1}')
        return [im1, im2]
    anim = animation.FuncAnimation(fig, update, frames=max_time_steps, interval=500, blit=True)
    anim.save('congestion_comparison.gif', writer='pillow', fps=2)
    plt.close(fig)
    print("Congestion comparison animation saved as 'congestion_comparison.gif'")

########################## NEW CODE: computed latency function components 


def get_line(start, end):
    """
    Bresenham's line algorithm 
    
    Args:
        start: Tuple (x1, y1) of starting position
        end: Tuple (x2, y2) of ending position
    
    Returns:
        List of (x, y) representing the path
    """
    x1, y1 = start
    x2, y2 = end
    points = []
    
    points = [(x1, y1)]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            prev_x, prev_y = x, y
            err -= dy
            if err < 0:
                points.append((prev_x, y + sy))
                y += sy
                err += dx
            x += sx
            points.append((x, y))
    else:
        err = dy / 2.0
        while y != y2:
            prev_x, prev_y = x, y
            err -= dx
            if err < 0:
                points.append((x + sx, prev_y))
                x += sx
                err += dy
            y += sy
            points.append((x, y))
            
    return points

def compute_H_matrix(simulation):
    grid_size = simulation.grid_size
    boundary_positions = simulation.get_boundary_positions()
    
    # Get indices of boundary positions for creating fully connected graph
    boundary_indices = [(i, j) for i, j in boundary_positions]
    num_boundaries = len(boundary_indices)
    
    # Create edges for fully connected graph between boundary points
    abstract_edges = []
    for i in range(num_boundaries):
        for j in range(i+1, num_boundaries):
            abstract_edges.append((boundary_indices[i], boundary_indices[j]))
    
    # Initialize H matrix (edges x grid_cells)
    num_edges = len(abstract_edges)
    H = np.zeros((num_edges, grid_size * grid_size))
    
    def is_adjacent(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1
        
    def on_same_boundary(p1, p2):
        return (p1[0] == p2[0] and (p1[0] == 0 or p1[0] == grid_size-1)) or \
               (p1[1] == p2[1] and (p1[1] == 0 or p1[1] == grid_size-1))
    
    for edge_idx, (start, end) in enumerate(abstract_edges):
        if is_adjacent(start, end):
            # adjacdent nodes only mark those two cells
            grid_idx_start = start[0] * grid_size + start[1]
            grid_idx_end = end[0] * grid_size + end[1]
            H[edge_idx, grid_idx_start] = 1
            H[edge_idx, grid_idx_end] = 1
            
        elif on_same_boundary(start, end):
            # if nodes are on same side of the subregion make a straight line
            if start[0] == end[0]:  # same row
                min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
                for y in range(min_y, max_y + 1):
                    grid_idx = start[0] * grid_size + y
                    H[edge_idx, grid_idx] = 1
            else:  # same column
                min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
                for x in range(min_x, max_x + 1):
                    grid_idx = x * grid_size + start[1]
                    H[edge_idx, grid_idx] = 1
                    
        else:
            # for any other case use the line algorithm
            line_points = get_line(start, end)
            for x, y in line_points:
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    grid_idx = x * grid_size + y
                    H[edge_idx, grid_idx] = 1
                
    return H, abstract_edges

def analyze_latency_components(simulation, predicted_flows, predicted_congestion, H, abstract_edges):
    grid_size = simulation.grid_size
            
    # Print raw model outputs
    #print("\nRaw Flows - Before scaling:")
    #print(f"Max value: {np.max(predicted_flows)}")
    #print(f"Min value: {np.min(predicted_flows)}")

    #print("\nRaw Congestion - Before scaling:")
    #print(f"Max value: {np.max(predicted_congestion)}")
    #print(f"Min value: {np.min(predicted_congestion)}")

    # Scale flows to [0,4] 
    predicted_flows_scaled = predicted_flows * 25.0
    #print("\nScaled Flows - Before max:")
    #print(f"Max value: {np.max(predicted_flows_scaled)}")
    #print(f"Min value: {np.min(predicted_flows_scaled)}")

    # Scale congestion to [1,5] (max 5x slowdown)
    predicted_congestion_scaled = 1 + (predicted_congestion * 2)
    #print("\nScaled Congestion - Before max:")
    #print(f"Max value: {np.max(predicted_congestion_scaled)}")
    #print(f"Min value: {np.min(predicted_congestion_scaled)}")

    # Take max over time for both
    x = np.zeros(grid_size * grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            x[i * grid_size + j] = np.max(predicted_flows_scaled[:, i, j])

    congestion_max = np.max(predicted_congestion_scaled, axis=0)
    
    # Compute l(x) using max congestion
    l_x = np.zeros(len(abstract_edges))
    for edge_idx, (start, end) in enumerate(abstract_edges):
        try:
            path = nx.shortest_path(simulation.graph, start, end)
            l_x[edge_idx] = sum(congestion_max[pos[0], pos[1]] for pos in path[:-1])
        except nx.NetworkXNoPath:
            l_x[edge_idx] = float('inf')
    
    # Compute b - free flow travel times (minimum 1 per cell)
    b = np.zeros(len(abstract_edges))
    for edge_idx, (start, end) in enumerate(abstract_edges):
        try:
            path = nx.shortest_path(simulation.graph, start, end)
            b[edge_idx] = len(path) - 1  # Each cell takes 1 unit of time minimum
        except nx.NetworkXNoPath:
            b[edge_idx] = float('inf')
    
    return l_x, x, b

def collect_single_sample(config, H, abstract_edges):
    simulation = SimplifiedSimulation(
        map_file=config["map_file"],
        time_steps=config["max_time_steps"],
        num_od_pairs=None  
    )
    
    try:
        predicted_flows, predicted_congestion, od_pairs, _ = simulation.run_simulation()
        l_x, x, b = analyze_latency_components(simulation, predicted_flows, predicted_congestion, H, abstract_edges)
            
        return (l_x, x, b)
    except Exception as e:
        print(f"Error in sample collection: {str(e)}")
        return None

def collect_and_fit(num_samples=3000):
    """Collect samples and fit A"""
    config = {
        "map_file": "empty-16-16.map",
        "max_time_steps": 31,
    }
    
    simulation = SimplifiedSimulation(
        map_file=config["map_file"],
        time_steps=config["max_time_steps"],
        num_od_pairs=None
    )
    
    # Compute H matrix once
    H, abstract_edges = compute_H_matrix(simulation)
    
    num_processes = max(1, os.cpu_count() - 1)  
    num_processes = 8
    print(f"Collecting {num_samples} samples using {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(
                partial(collect_single_sample, H=H, abstract_edges=abstract_edges),
                [config] * num_samples
            ),
            total=num_samples,
            desc="Collecting samples"
        ))
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError("No valid samples collected")
        
    all_l_x, all_x, all_b = zip(*valid_results)
    
    L = np.vstack(all_l_x)
    X = np.vstack(all_x)
    B = np.vstack(all_b)
    
    print("\nData shapes:")
    print(f"L shape: {L.shape}")
    print(f"X shape: {X.shape}")
    print(f"B shape: {B.shape}")
    print(f"H shape: {H.shape}")
    
    print("\nValue ranges after scaling:")
    print(f"L range: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"X range: [{np.min(X):.4f}, {np.max(X):.4f}]")
    print(f"B range: [{np.min(B):.4f}, {np.max(B):.4f}]")
    
    print("\nComputing matrix products...")
    HX = X @ H.T
    
    #print("\nSolving least squares problem...")
    A, residuals, rank, s = np.linalg.lstsq(HX, L - B, rcond=None)

    #A = (A + A.T) / 2
    
    print("\nFitting completed!")
    print(f"A matrix shape: {A.shape}")
    print(f"A matrix statistics:")
    print(f"Mean value: {np.mean(A):.4f}")
    print(f"Max value: {np.max(A):.4f}")
    print(f"Min value: {np.min(A):.4f}")
    
    return A, H, (L, X, B)

def main():

    A, H, (L, X, B) = collect_and_fit(num_samples=10000)

    save_path = f"results_.npz"
    np.savez(save_path,
             A=A,
             H=H,
             L=L,
             X=X,
             B=B)
    
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()