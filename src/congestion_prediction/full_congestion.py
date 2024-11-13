import numpy as np
import torch
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Import from temporal_flow
from nn import MAPFCongestionModel as FlowModel
from utils import get_device, set_seed

# Import from temporal_congestion
from nn import MAPFCongestionModel as CongestionModel

from cbs import conflict_based_search, MAPFProblem, Environment

""""
Evaluates the congestion and flow prediction models in a realistic scenario (doesn't rely on preprocessed data): 
1. generate random OD pairs 
2. compute basic shortest path information as input to the flow model 
3. uses predicted flows as input to the congestion prediction model 
4. predicts the temporal congestion across the grid
5. visualizes the temporal congestion/flow prediction

this script forms the basis for the congestion prediction in the full simulation

"""

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
            "max_time_steps": self.time_steps,
            "dropout_rate": 0.2,
        }
        model = FlowModel(**config).to(self.device)
        model_path = "models/best_flow_model_60.pth"
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
            "max_time_steps": self.time_steps,
            "dropout_rate": 0.2,
        }
        model = CongestionModel(**config).to(self.device)
        model_path = "models/best_congestion_model_60.pth"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    def create_graph(self):
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        # Remove edges for non-traversable cells
        edges_to_remove = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:  # Non-traversable cell
                    edges_to_remove.extend([((i, j), n) for n in G.neighbors((i, j))])
        G.remove_edges_from(edges_to_remove)
        return G
    
    def get_boundary_positions(self):
        height, width = self.grid_size, self.grid_size
        boundary = []
        for i in range(height):
            if self.grid[i, 0] == 1:
                boundary.append((i, 0))
            if self.grid[i, width - 1] == 1:
                boundary.append((i, width - 1))
        for j in range(1, width - 1):
            if self.grid[0, j] == 1:
                boundary.append((0, j))
            if self.grid[height - 1, j] == 1:
                boundary.append((height - 1, j))
        return boundary
    
    def generate_od_pairs(self):
        boundary_positions = self.get_boundary_positions()
        random.shuffle(boundary_positions)
        
        n_agents = random.randint(10, 49)  # Using Python's built-in random
        
        if len(boundary_positions) < 2 * n_agents:
            n_agents = len(boundary_positions) // 2  # Ensure we have enough positions
        
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
            node_features[source[0], start_idx, 0] = 1  # Source
            node_features[target[0], end_idx, 1] = 1  # Destination
            
            path = nx.shortest_path(self.graph, (source[1], source[2]), (target[1], target[2]))
            
            # Add flow along the path
            for t, (x, y) in enumerate(path):
                if t < self.time_steps:
                    node_idx = x * self.grid_size + y
                    node_features[t, node_idx, 2] += 1  # Flow
        
        edge_index = self.create_edge_index()
        
        return Data(x=node_features.to(self.device), edge_index=edge_index.to(self.device))
    
    def create_edge_index(self):
        edges = list(self.graph.edges())
        edge_index = torch.tensor([[self.grid_size * x1 + y1, self.grid_size * x2 + y2] for (x1, y1), (x2, y2) in edges], dtype=torch.long).t().contiguous()
        return edge_index
    
    def predict_flows(self, od_pairs):
        input_data = self.create_flow_input_data(od_pairs)
        
        with torch.no_grad():
            predicted_flows = self.flow_model(input_data)
        
        # Scale up the predicted flows
        scaling_factor = 3.0
        predicted_flows *= scaling_factor
        
        return predicted_flows.view(self.time_steps, self.grid_size, self.grid_size).squeeze().cpu().numpy()
    
    def create_congestion_input_data(self, predicted_flows):
        num_nodes = self.grid_size * self.grid_size
        node_features = torch.zeros((self.time_steps, num_nodes, 3))
        
        # Set the flow feature (index 2) to the predicted flows
        node_features[:, :, 2] = torch.tensor(predicted_flows.reshape(self.time_steps, -1))
        
        edge_index = self.create_edge_index()
        
        return Data(x=node_features.to(self.device), edge_index=edge_index.to(self.device))
    
    def predict_congestion(self, predicted_flows):
        input_data = self.create_congestion_input_data(predicted_flows)

        with torch.no_grad():
            predicted_congestion = self.congestion_model(input_data)
        congestion = predicted_congestion.view(self.time_steps, self.grid_size, self.grid_size).squeeze().cpu().numpy()
        congestion = np.maximum(congestion, 0)
        return congestion
    
    def run_simulation(self):
        # Generate OD pairs
        od_pairs = self.generate_od_pairs()
        
        # Predict flows
        predicted_flows = self.predict_flows(od_pairs)
        
        # Predict congestion
        predicted_congestion = self.predict_congestion(predicted_flows)
        
        return predicted_flows, predicted_congestion, od_pairs

def visualize_congestion(congestion, config):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        im = ax.imshow(congestion[frame], cmap='YlOrRd', vmin=0, vmax=congestion.max()) # red: YlOrRd
        ax.set_title(f'Predicted Congestion at Time Step {frame + 1}')
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=config['max_time_steps'], interval=500, blit=True)
    anim.save('congestion_animation.gif', writer='pillow', fps=2)
    plt.close(fig)
    print("Congestion animation saved as 'congestion_animation.gif'")
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
    
    # Normalize congestion
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
    
    # Solve using CBS
    solution = conflict_based_search(problem)
    
    if solution is None:
        raise ValueError("CBS couldn't find a solution for the given problem.")
    
    # Compute ground truth congestion
    ground_truth_congestion = compute_ground_truth_congestion(env, goal_pos, solution)
    
    return ground_truth_congestion

def visualize_congestion_comparison(predicted_congestion, ground_truth_congestion, config):
    max_time_steps = min(predicted_congestion.shape[0], ground_truth_congestion.shape[0])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        im1 = ax1.imshow(predicted_congestion[frame], cmap='YlOrRd', vmin=0, vmax=1)
        ax1.set_title(f'Predicted Congestion at Time Step {frame + 1}')
        
        im2 = ax2.imshow(ground_truth_congestion[frame], cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title(f'Ground Truth Congestion at Time Step {frame + 1}')
        
        diff = np.abs(predicted_congestion[frame] - ground_truth_congestion[frame])
        im3 = ax3.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax3.set_title(f'Absolute Difference at Time Step {frame + 1}')
        
        return [im1, im2, im3]
    
    anim = animation.FuncAnimation(fig, update, frames=max_time_steps, interval=500, blit=True)
    anim.save('congestion_comparison.gif', writer='pillow', fps=2)
    plt.close(fig)
    print("Congestion comparison animation saved as 'congestion_comparison.gif'")

def main():
    config = {
        "map_file": "empty-32-32.map",
        "max_time_steps": 63,
    }
    
    simulation = SimplifiedSimulation(
        map_file=config["map_file"],
        time_steps=config["max_time_steps"],
        num_od_pairs=None  
    )
    
    try:
        predicted_flows, predicted_congestion, od_pairs = simulation.run_simulation()
        print("\nSimulation completed successfully.")

        # Visualize predicted congestion
        visualize_congestion(predicted_congestion, config)
        print("Agents: ", len(od_pairs))
        # Compute ground truth congestion
        ground_truth_congestion = get_ground_truth_congestion(
            config["map_file"],
            len(od_pairs),
            config["max_time_steps"],
            od_pairs
        )
        
        # Visualize comparison between predicted and ground truth congestion
        visualize_congestion_comparison(predicted_congestion, ground_truth_congestion, config)

        # Print OD pairs
        print("\nGenerated OD pairs:")
        for source, target in od_pairs:
            print(f"From {source[1:]} to {target[1:]}")

    except Exception as e:
        print(f"An error occurred during simulation: {str(e)}")

if __name__ == "__main__":
    main()