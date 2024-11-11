from typing import Tuple, List, Set, Optional
import networkx as nx
import numpy as np

class Environment:
    FREE, OBSTACLE, AGENT, GOAL = range(4)
    
    def __init__(self):
        self.size: Tuple[int, int] = None
        self.graph: nx.Graph = None
        
    def contains_node(self, pos: Tuple[int, int]) -> bool:
        raise NotImplementedError
        
    def contains_edge(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        raise NotImplementedError
        
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        raise NotImplementedError
        
    def get_state(self) -> np.ndarray:
        raise NotImplementedError

class GridWorld(Environment):
    def __init__(self, size: Tuple[int, int], obstacles: List[Tuple[int, int]] = None):
        super().__init__()
        self.size = size
        self.obstacles = set(obstacles) if obstacles else set()
        self.graph = self._create_grid_graph()
        
    def _create_grid_graph(self) -> nx.Graph:
        G = nx.grid_2d_graph(*self.size)
        for obs in self.obstacles:
            if obs in G:
                G.remove_node(obs)
        return G
        
    def contains_node(self, pos: Tuple[int, int]) -> bool:
        return pos in self.graph
        
    def contains_edge(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        return self.graph.has_edge(pos1, pos2)
        
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        return list(self.graph.neighbors(pos))
        
    def get_state(self) -> np.ndarray:
        state = np.zeros(self.size, dtype=int)
        for obs in self.obstacles:
            state[obs] = self.OBSTACLE
        return state

    def is_valid_path(self, path: List[Tuple[int, int]]) -> bool:
        if not path:
            return True
            
        for pos in path:
            if not self.contains_node(pos):
                return False
                
        for i in range(len(path) - 1):
            if not self.contains_edge(path[i], path[i+1]):
                return False
                
        return True