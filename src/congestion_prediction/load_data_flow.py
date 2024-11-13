import os
import pickle
from collections import namedtuple
from typing import Dict, List, Tuple
from queue import PriorityQueue

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from utils import dijkstra_shortest_path, get_device


"""
loads the generated data to be used in the prediciton model
"""

MAPFInstance = namedtuple('MAPFInstance', ['problem_instance', 'n_agents', 'goals', 'path_lengths', 'paths'])

def load_data_from_folder(folder_path: str) -> Dict[str, Dict]:
    data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'rb') as f:
                    file_data = pickle.load(f)
                    data[file_path] = file_data
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
    return data

def load_data_from_sub_folders(parent_folder: str) -> Dict[str, Dict]:
    all_data = {}
    for folder_name in tqdm(os.listdir(parent_folder), desc="Loading data from files..."):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            folder_data = load_data_from_folder(folder_path)
            all_data.update(folder_data)
    return all_data

def mapf_dataset(all_data: Dict[str, Dict]) -> List[MAPFInstance]:
    return [
        MAPFInstance(
            file_path,
            len(data['problem'].env.agent_pos),
            data['problem'].goals,
            data['solution'].path_lengths(),
            [[vertex.pos for vertex in path.vertexes] for path in data['solution'].paths]
        )
        for file_path, data in all_data.items()
        if 'problem' in data and 'solution' in data
    ]

def get_max_makespan(dataset: List[MAPFInstance]) -> int:
    return max(max(len(path) for path in instance.paths) for instance in dataset)

def get_time_augmented_features_and_labels(instance: MAPFInstance, grid_size: int, max_time_steps: int, dense_matrix: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    num_nodes = grid_size * grid_size
    node_features = torch.zeros((max_time_steps, num_nodes, 3))
    flow_labels = torch.zeros((max_time_steps, num_nodes))
    
    # Compute shortest paths and set features
    for agent_idx, goal in enumerate(instance.goals):
        start = instance.paths[agent_idx][0]
        end = goal
        start_idx = start[0] * grid_size + start[1]
        end_idx = end[0] * grid_size + end[1]
        
        node_features[:, start_idx, 0] = 1  # Source
        node_features[:, end_idx, 1] = 1  # Destination
        
        path = dijkstra_shortest_path(dense_matrix, start, end)
        for t, node in enumerate(path):
            if t < max_time_steps:
                node_idx = node[0] * grid_size + node[1]
                node_features[t, node_idx, 2] += 1  # Flow
    
    # Compute actual flow for labels
    for t in range(min(max_time_steps, max(len(path) for path in instance.paths))):
        for path in instance.paths:
            if t < len(path):
                node_idx = path[t][0] * grid_size + path[t][1]
                flow_labels[t, node_idx] += 1
    
    # Normalize flow labels
    max_flow = flow_labels.max()
    if max_flow > 0:
        flow_labels /= max_flow
    
    return node_features, flow_labels

def create_adj_matrix(dense_matrix: List[List[int]]) -> Dict[int, List[int]]:
    n = len(dense_matrix)
    adj_list_dict = {i: [] for i in range(n*n)}
    
    for i in range(n):
        for j in range(n):
            if dense_matrix[i][j] == 0:
                current_node = i * n + j
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and dense_matrix[ni][nj] == 0:
                        neighbor_node = ni * n + nj
                        adj_list_dict[current_node].append(neighbor_node)
                        adj_list_dict[neighbor_node].append(current_node)
    
    return {k: list(set(v)) for k, v in adj_list_dict.items()}

def build_edge_index(adjacency_list_dict: Dict[int, List[int]], num_of_nodes: int, add_self_edges: bool = True) -> torch.Tensor:
    edges = [(src, trg) for src, trgs in adjacency_list_dict.items() for trg in trgs]
    if add_self_edges:
        edges.extend((i, i) for i in range(num_of_nodes))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def load_data(data_folder: str, grid_size: int = 8) -> Tuple[List[Data], int]:
    device = get_device()
    print("Using Device: ", device)
    
    all_data = load_data_from_sub_folders(data_folder)
    dataset = mapf_dataset(all_data)
    
    print(f"Number of instances in dataset: {len(dataset)}")
    
    max_time_steps = get_max_makespan(dataset)
    
    if not all_data:
        raise ValueError("No data found in the specified folder.")
    
    # Create an empty 32x32 grid
    dense_matrix = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    adj_list_dict = create_adj_matrix(dense_matrix)
    edge_index = build_edge_index(adj_list_dict, grid_size ** 2).to(device)
    
    pyg_dataset = []
    for instance in tqdm(dataset, desc="Loading data features...", unit="instance"):
        try:
            node_features, flow_labels = get_time_augmented_features_and_labels(instance, grid_size, max_time_steps, dense_matrix)
            pyg_data = Data(x=node_features.to(device), edge_index=edge_index, y=flow_labels.to(device))
            pyg_dataset.append(pyg_data)
        except Exception as e:
            print(f"Error processing instance {instance.problem_instance}: {e}")
    
    return pyg_dataset, max_time_steps

if __name__ == "__main__":
    data_folder = "data/8x8/test"
    pyg_dataset, max_time_steps = load_data(data_folder)
    print(f"Loaded {len(pyg_dataset)} instances with max time steps: {max_time_steps}")
    print(f"First instance features shape: {pyg_dataset[0].x.shape}")
    print(f"First instance labels shape: {pyg_dataset[0].y.shape}")