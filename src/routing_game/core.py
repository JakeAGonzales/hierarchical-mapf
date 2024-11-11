from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
import numpy as np
import time
from dataclasses import dataclass

from ..common import get_canonical_edge, dijkstra_shortest_path
from .flow import frank_wolfe_step, normalize_flows, calculate_edge_costs
from .od_pairs import generate_od_pairs

@dataclass
class RoutingGameConfig:
    grid_size: int = 32
    subregion_size: int = 8
    num_od_pairs: int = 10
    total_flow: float = 100.0
    boundary_type: str = "full_grid"  # or "all_boundaries"

class AbstractedRoutingGame:
    def __init__(self, config: Optional[RoutingGameConfig] = None):
        if config is None:
            config = RoutingGameConfig()
            
        self.config = config
        self.grid_size = self.config.grid_size
        self.subregion_size = self.config.subregion_size
        self.num_od_pairs = self.config.num_od_pairs
        self.total_flow = self.config.total_flow
        self.num_subregions = self.config.grid_size // self.config.subregion_size
        self.boundary_type = self.config.boundary_type
        
        if self.boundary_type not in ["full_grid", "all_boundaries"]:
            raise ValueError("boundary_type must be either 'full_grid' or 'all_boundaries'")
        
        self.graph = self._create_abstracted_graph()
        self.od_pairs, self.demands = generate_od_pairs(self)
        self.edge_flows = {get_canonical_edge(*edge): 0 for edge in self.graph.edges()}
        self.edge_costs = {get_canonical_edge(*edge): 1 for edge in self.graph.edges()}

    def _create_abstracted_graph(self) -> nx.Graph:
        G = nx.Graph()
        
        # Add boundary nodes and create fully connected subregions
        for i in range(self.num_subregions):
            for j in range(self.num_subregions):
                boundary_positions = self._get_subregion_boundary_positions(i, j)
                G.add_nodes_from(boundary_positions)
                
                # Create fully connected subregion
                for pos1 in boundary_positions:
                    for pos2 in boundary_positions:
                        if pos1 < pos2:  # Avoid self-loops and duplicates
                            G.add_edge(pos1, pos2)
        
        # Connect adjacent subregions through aligned positions
        self._connect_adjacent_subregions(G)
        
        return G

    def _connect_adjacent_subregions(self, G: nx.Graph):
        for i in range(self.num_subregions):
            for j in range(self.num_subregions):
                current_positions = self._get_subregion_boundary_positions(i, j)
                
                # Connect to right subregion
                if j < self.num_subregions - 1:
                    self._connect_horizontal_regions(G, i, j, current_positions)
                
                # Connect to bottom subregion
                if i < self.num_subregions - 1:
                    self._connect_vertical_regions(G, i, j, current_positions)

    def _connect_horizontal_regions(self, G: nx.Graph, i: int, j: int, current_positions: List[Tuple[int, int]]):
        right_positions = self._get_subregion_boundary_positions(i, j + 1)
        for pos in current_positions:
            if pos[1] == (j + 1) * self.subregion_size - 1:  # Right edge
                aligned_pos = (pos[0], pos[1] + 1)
                if aligned_pos in right_positions:
                    G.add_edge(pos, aligned_pos)

    def _connect_vertical_regions(self, G: nx.Graph, i: int, j: int, current_positions: List[Tuple[int, int]]):
        bottom_positions = self._get_subregion_boundary_positions(i + 1, j)
        for pos in current_positions:
            if pos[0] == (i + 1) * self.subregion_size - 1:  # Bottom edge
                aligned_pos = (pos[0] + 1, pos[1])
                if aligned_pos in bottom_positions:
                    G.add_edge(pos, aligned_pos)

    def _get_subregion_boundary_positions(self, region_i: int, region_j: int) -> List[Tuple[int, int]]:
        start_i = region_i * self.subregion_size
        start_j = region_j * self.subregion_size
        end_i = start_i + self.subregion_size
        end_j = start_j + self.subregion_size
        
        boundary_positions = []
        
        # Top and bottom edges
        for j in range(start_j, end_j):
            boundary_positions.append((start_i, j))  # Top
            boundary_positions.append((end_i - 1, j))  # Bottom
            
        # Left and right edges (excluding corners)
        for i in range(start_i + 1, end_i - 1):
            boundary_positions.append((i, start_j))  # Left
            boundary_positions.append((i, end_j - 1))  # Right
            
        return boundary_positions

    def run_frank_wolfe(self, max_iterations: int = 100, 
                       convergence_threshold: float = 1e-3,
                       sample_rate: int = 1):
        start_time = time.time()
        costs = []
        all_flows = []
        
        for iteration in range(max_iterations):
            new_flows = frank_wolfe_step(self)
            step_size = 2 / (iteration + 2)
            
            max_diff = self._update_flows(new_flows, step_size)
            self.edge_costs = calculate_edge_costs(self.edge_flows)
            
            current_cost = self.total_system_cost()
            costs.append(current_cost)
            
            if iteration % sample_rate == 0:
                normalized_flows = normalize_flows(self.edge_flows)
                all_flows.append((iteration + 1, normalized_flows))
            
            if max_diff < convergence_threshold:
                break
        
        return (normalize_flows(self.edge_flows), costs, 
                time.time() - start_time, all_flows)

    def _update_flows(self, new_flows: Dict, step_size: float) -> float:
        max_diff = 0
        for edge in self.edge_flows.keys():
            old_flow = self.edge_flows[edge]
            new_flow = (1 - step_size) * old_flow + step_size * new_flows.get(edge, 0)
            self.edge_flows[edge] = new_flow
            max_diff = max(max_diff, abs(new_flow - old_flow))
        return max_diff

    def total_system_cost(self) -> float:
        return sum(self.edge_costs[edge] * flow 
                  for edge, flow in self.edge_flows.items())