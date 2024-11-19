import networkx as nx
import numpy as np
from typing import List, Tuple, Set
import copy

class Environment:
    FREE, OBSTACLE, AGENT, GOAL, GOAL_REACHED = range(5)

    def __init__(self, size, obstacle_pos, agent_pos):
        self.size = size
        self.obstacle_pos = set(obstacle_pos)
        self.agent_pos = agent_pos
        self.agents = list(range(len(agent_pos)))
        self.graph = self._create_graph()
        self.data = {}
        for cartesian_index in obstacle_pos:
            self.data[cartesian_index] = self.OBSTACLE
        for i, cartesian_index in enumerate(agent_pos):
            self.data[cartesian_index] = self.AGENT

    def _create_graph(self):
        G = nx.grid_2d_graph(*self.size)
        for obs in self.obstacle_pos:
            G.remove_node(obs)
        return G

    def get_obstacles(self):
        return copy.deepcopy(self.obstacle_pos)
    
    def get_agents(self):
        return copy.deepcopy(self.agent_pos)

    def update_agent_pos(self, ids, positions):
        for i, j in enumerate(ids):
            if self.agent_pos[j] in self.data:
                del self.data[self.agent_pos[j]]
            self.data[positions[i]] = self.AGENT
            self.agent_pos[j] = positions[i]

    def dense_matrix(self):
        mat = np.zeros(self.size, dtype=int)
        for pos in self.data:
            mat[pos] = self.data[pos]
        return mat
    

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