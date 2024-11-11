from typing import List, Tuple, Dict, Set
import networkx as nx

from ..common import Environment, GridWorld
from .constraints import RegionActionGenerator

class GridRegion:
    def __init__(self, grid_world: GridWorld, location: Tuple[int, int], size: Tuple[int, int]):
        self.size = size
        self.location = location
        self.boundary = []
        nodes = []
        
        for node in grid_world.G.nodes:
            row_in_bounds = location[0] <= node[0] < location[0] + size[0]
            col_in_bounds = location[1] <= node[1] < location[1] + size[1]
            
            if row_in_bounds and col_in_bounds:
                nodes.append(node)
                # Add to boundary if on any edge
                if (node[0] == location[0] or 
                    node[0] == location[0] + size[0] - 1 or
                    node[1] == location[1] or 
                    node[1] == location[1] + size[1] - 1):
                    self.boundary.append(node)

        self.G = nx.subgraph(grid_world.G, nodes)

    def contains_node(self, u: Tuple[int, int]) -> bool:
        return u in self.G.nodes
    
    def contains_edge(self, u: Tuple[int, int], v: Tuple[int, int]) -> bool:
        return (u, v) in self.G.edges

class RegionalEnvironment(Environment):
    def __init__(self, gridworld: GridWorld, region_graph: nx.Graph):
        self.gridworld = gridworld
        self.region_graph = region_graph
        self.action_generators = {}
        
        for region_id in self.region_graph.nodes:
            region = self.region_graph.nodes[region_id]['env']
            self.action_generators[region_id] = RegionActionGenerator(
                gridworld, region
            )

    def contains_node(self, u: Tuple[int, int]) -> bool:
        return self.gridworld.contains_node(u)
    
    def contains_edge(self, u: Tuple[int, int], v: Tuple[int, int]) -> bool:
        return self.gridworld.contains_edge(u, v)
    
    def dense_matrix(self):
        return self.gridworld.dense_matrix()

class SimpleRegionalEnvironment(RegionalEnvironment):
    def __init__(self, world_size: Tuple[int, int], region_size: Tuple[int, int], 
                 obstacles: List[Tuple[int, int]] = None):
        self.world_size = world_size
        self.region_size = region_size
        self.gridworld = GridWorld(world_size, obstacles or [])
        
        n_rows = world_size[0] // region_size[0]
        n_cols = world_size[1] // region_size[1]
        
        # Create and connect regions
        self.region_graph = self._create_region_graph(n_rows, n_cols)
        super().__init__(self.gridworld, self.region_graph)

    def _create_region_graph(self, n_rows: int, n_cols: int) -> nx.Graph:
        G = nx.Graph()
        
        # Create regions
        for i in range(n_rows):
            for j in range(n_cols):
                region = GridRegion(
                    self.gridworld,
                    (i * self.region_size[0], j * self.region_size[1]),
                    self.region_size
                )
                G.add_node((i, j), env=region)
                
                # Connect to existing neighbors
                if i > 0:  # Connect to region above
                    edges = self._find_boundary_edges(
                        region, G.nodes[(i-1, j)]['env']
                    )
                    G.add_edge((i, j), (i-1, j), boundary=edges)
                
                if j > 0:  # Connect to region to left
                    edges = self._find_boundary_edges(
                        region, G.nodes[(i, j-1)]['env']
                    )
                    G.add_edge((i, j), (i, j-1), boundary=edges)
        
        return G

    def _find_boundary_edges(self, region1: GridRegion, 
                           region2: GridRegion) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        edges = []
        for u in region1.boundary:
            for v in region2.boundary:
                if self.gridworld.contains_edge(u, v):
                    edges.append((u, v))
        return edges