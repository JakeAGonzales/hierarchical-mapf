from typing import List, Dict, Tuple
import copy

from .environment import Environment
from .constraints import PathVertex

class Path:
    def __init__(self, vertexes: List[PathVertex], cost: float):
        self.vertexes = vertexes
        self.cost = cost

    def __getitem__(self, i: int) -> PathVertex:
        return self.vertexes[i]

    def __len__(self) -> int:
        return len(self.vertexes)

class MAPFProblem:
    def __init__(self, env: Environment, goals: List[Tuple[int, int]]):
        if len(goals) != len(env.agent_pos):
            raise ValueError("Goal states must match number of agents in environment")
        self.n_agents = len(goals)
        self.env = env
        self.goals = goals

    @property 
    def agent_positions(self) -> List[Tuple[int, int]]:
        return self.env.agent_pos

class MAPFSolution:
    def __init__(self, paths: List[Path]):
        self.paths = paths
        self.makespan = max(len(path) for path in paths)
        
    def path_lengths(self) -> List[int]:
        return [len(p) for p in self.paths]
    
    def total_cost(self) -> float:
        return sum(len(path) for path in self.paths)
    
    def get_state_at_time(self, t: int) -> Dict[int, Tuple[int, int]]:
        positions = {}
        for agent_id, path in enumerate(self.paths):
            if t < len(path):
                positions[agent_id] = path[t].pos
            else:
                positions[agent_id] = path[-1].pos
        return positions

    def validate(self) -> bool:
        # Check for collisions between agents
        for t in range(self.makespan):
            positions = self.get_state_at_time(t)
            
            # Check vertex conflicts
            seen_positions = {}
            for agent_id, pos in positions.items():
                if pos in seen_positions:
                    return False
                seen_positions[pos] = agent_id
                
            # Check edge conflicts
            if t > 0:
                prev_positions = self.get_state_at_time(t-1)
                for agent1 in range(len(self.paths)):
                    for agent2 in range(agent1 + 1, len(self.paths)):
                        if (positions[agent1] == prev_positions[agent2] and 
                            positions[agent2] == prev_positions[agent1]):
                            return False
        
        return True