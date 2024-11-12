# core.py

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Set
import networkx as nx
import time
import random
from collections import defaultdict
from ..common.search import dijkstra_shortest_path

from ..common import get_canonical_edge

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
        self.grid_size = config.grid_size
        self.subregion_size = config.subregion_size
        self.num_od_pairs = config.num_od_pairs
        self.total_flow = config.total_flow
        self.num_subregions = config.grid_size // config.subregion_size
        self.boundary_type = config.boundary_type
        
        if self.boundary_type not in ["full_grid", "all_boundaries"]:
            raise ValueError("boundary_type must be either 'full_grid' or 'all_boundaries'")
        
        self.graph = self._create_abstracted_graph()
        self.od_pairs, self.demands = self._generate_od_pairs()
        self.edge_flows = {get_canonical_edge(*edge): 0 for edge in self.graph.edges()}
        self.edge_costs = {get_canonical_edge(*edge): 1 for edge in self.graph.edges()}

    def edge_cost(self, edge, flow: float) -> float:
        """Calculate cost for a single edge given its flow"""
        return 1 + 0.5 * flow

    def update_edge_costs(self) -> None:
        """Update all edge costs based on current flows"""
        for edge, flow in self.edge_flows.items():
            self.edge_costs[edge] = self.edge_cost(edge, flow)

    def total_system_cost(self) -> float:
        """Calculate total system cost"""
        return sum(self.edge_costs[edge] * flow for edge, flow in self.edge_flows.items())

    def _get_subregion_boundary_positions(self, region_i: int, region_j: int) -> List[Tuple[int, int]]:
        """Get boundary positions for a subregion"""
        start_i = region_i * self.subregion_size
        start_j = region_j * self.subregion_size
        end_i = start_i + self.subregion_size
        end_j = start_j + self.subregion_size
        
        boundary_positions = []
        
        # Top and bottom edges
        for j in range(start_j, end_j):
            boundary_positions.append((start_i, j))
            boundary_positions.append((end_i - 1, j))
            
        # Left and right edges (excluding corners)
        for i in range(start_i + 1, end_i - 1):
            boundary_positions.append((i, start_j))
            boundary_positions.append((i, end_j - 1))
            
        return boundary_positions

    def _create_abstracted_graph(self) -> nx.Graph:
        """Create the abstracted graph with connected subregions"""
        G = nx.Graph()
        
        # First add all boundary positions and create fully connected subregions
        for i in range(self.num_subregions):
            for j in range(self.num_subregions):
                boundary_positions = self._get_subregion_boundary_positions(i, j)
                G.add_nodes_from(boundary_positions)
                
                # Create fully connected graph within subregion
                for pos1 in boundary_positions:
                    for pos2 in boundary_positions:
                        if pos1 < pos2:  # Avoid self-loops and duplicate edges
                            G.add_edge(pos1, pos2)
        
        # Connect adjacent subregions
        self._connect_adjacent_regions(G)
        return G

    def _connect_adjacent_regions(self, G: nx.Graph) -> None:
        """Connect adjacent subregions through aligned positions"""
        for i in range(self.num_subregions):
            for j in range(self.num_subregions):
                current_positions = self._get_subregion_boundary_positions(i, j)
                
                # Connect to right subregion
                if j < self.num_subregions - 1:
                    right_positions = self._get_subregion_boundary_positions(i, j + 1)
                    for pos in current_positions:
                        if pos[1] == (j + 1) * self.subregion_size - 1:  # Right edge
                            aligned_pos = (pos[0], pos[1] + 1)
                            if aligned_pos in right_positions:
                                G.add_edge(pos, aligned_pos)
                
                # Connect to bottom subregion
                if i < self.num_subregions - 1:
                    bottom_positions = self._get_subregion_boundary_positions(i + 1, j)
                    for pos in current_positions:
                        if pos[0] == (i + 1) * self.subregion_size - 1:  # Bottom edge
                            aligned_pos = (pos[0] + 1, pos[1])
                            if aligned_pos in bottom_positions:
                                G.add_edge(pos, aligned_pos)

    def _get_all_boundary_nodes(self) -> List[Tuple[int, int]]:
        """Get all boundary nodes based on boundary type"""
        if self.boundary_type == "full_grid":
            return self._get_grid_boundary_positions()
        else:  # all_boundaries
            nodes = []
            for i in range(self.num_subregions):
                for j in range(self.num_subregions):
                    nodes.extend(self._get_subregion_boundary_positions(i, j))
            return list(set(nodes))  # Remove duplicates

    def _get_grid_boundary_positions(self) -> List[Tuple[int, int]]:
        """Get boundary positions for full grid"""
        boundary = []
        # Top and bottom edges
        for j in range(self.grid_size):
            boundary.append((0, j))
            boundary.append((self.grid_size - 1, j))
        # Left and right edges (excluding corners)
        for i in range(1, self.grid_size - 1):
            boundary.append((i, 0))
            boundary.append((i, self.grid_size - 1))
        return boundary

    def _generate_od_pairs(self) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], Dict]:
        """Generate origin-destination pairs and their demands"""
        available_positions = self._get_all_boundary_nodes()
        n_starts = min(self.num_od_pairs, len(available_positions) // 2)
        
        remaining_positions = available_positions.copy()
        random.shuffle(remaining_positions)
        origins = remaining_positions[:n_starts]
        
        for pos in origins:
            remaining_positions.remove(pos)
            
        random.shuffle(remaining_positions)
        destinations = remaining_positions[:n_starts]
        
        od_pairs = [(start, end) for start, end in zip(origins, destinations)]
        demands = {pair: self.total_flow for pair in od_pairs}
        
        return od_pairs, demands

    def run_frank_wolfe(self, max_iterations: int = 100, 
                       convergence_threshold: float = 1e-3,
                       sample_rate: int = 1):
        """Run Frank-Wolfe algorithm to find equilibrium flows"""
        start_time = time.time()
        costs = []
        all_flows = []
        
        for iteration in range(max_iterations):
            new_flows = self._frank_wolfe_step()
            
            step_size = 2 / (iteration + 2)
            max_diff = 0
            
            for edge in self.edge_flows.keys():
                old_flow = self.edge_flows[edge]
                new_flow = (1 - step_size) * old_flow + step_size * new_flows.get(edge, 0)
                self.edge_flows[edge] = new_flow
                max_diff = max(max_diff, abs(new_flow - old_flow))
            
            self.update_edge_costs()
            current_cost = self.total_system_cost()
            costs.append(current_cost)
            
            if iteration % sample_rate == 0:
                normalized_flows = self.normalize_flows()
                all_flows.append((iteration + 1, normalized_flows))
            
            print(f"Iteration {iteration + 1}: Total System Cost = {current_cost:.4f}")
            
            if max_diff < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self.normalize_flows(), costs, time.time() - start_time, all_flows

    def _frank_wolfe_step(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """Calculate new flows using the Frank-Wolfe algorithm step"""
        new_flows = defaultdict(float)
        for od_pair in self.od_pairs:
            start, end = od_pair
            path = dijkstra_shortest_path(self.graph, start, end, self.edge_costs)
            if path:
                demand = self.demands[od_pair]
                for i in range(len(path) - 1):
                    edge = get_canonical_edge(path[i], path[i+1])
                    new_flows[edge] += demand
        return new_flows

    def normalize_flows(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
        """Normalize flow values to be between 0 and 1"""
        max_flow = max(self.edge_flows.values())
        if max_flow > 0:
            return {edge: flow / max_flow for edge, flow in self.edge_flows.items()}
        return self.edge_flows