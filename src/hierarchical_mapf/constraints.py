from typing import Dict, Set, Generator, Tuple, Any
from dataclasses import dataclass

from ..common import GridWorld, PathVertex, PathEdge, manhattan_distance

class RegionActionGenerator:
    def __init__(self, world: GridWorld, region: Any, constraints: Dict = None):
        self.world = world
        self.region = region
        self.constraints = constraints or {}

    def actions(self, v: PathVertex) -> Generator[Tuple[PathVertex, PathEdge], None, None]:
        if not self.region.contains_node(v.pos):
            return
            
        for pos in self.world.G.adj[v.pos]:
            u = PathVertex(pos, v.t + 1)
            e = PathEdge(v.pos, pos, v.t)
            
            if u in self.constraints:
                continue
            if e in self.constraints:
                continue
            if e.compliment() in self.constraints:
                continue
                
            yield (u, e)

class LocationGoal:
    def __init__(self, pos: Tuple[int, int]):
        self.pos = pos

    def heuristic(self, pos: Tuple[int, int]) -> float:
        return manhattan_distance(pos, self.pos)
        
    def satisfied(self, pos: Tuple[int, int]) -> bool:
        return pos == self.pos

class SetGoal:
    def __init__(self, positions: Set[Tuple[int, int]]):
        self.positions = frozenset(positions)

    def heuristic(self, pos: Tuple[int, int]) -> float:
        return min(manhattan_distance(pos, p) for p in self.positions)
        
    def satisfied(self, pos: Tuple[int, int]) -> bool:
        return pos in self.positions

class BoundaryGoal:
    def __init__(self, env: Any, source: Tuple[int, int], dest: Tuple[int, int], 
                 final_goal: Tuple[int, int]):
        if (source, dest) not in env.region_graph.edges:
            raise ValueError(f"source {source} and dest {dest} are not connected")
            
        edges = env.region_graph.edges[source, dest]['boundary']
        region = env.region_graph.nodes[source]['env']
        nodes = [v for e in edges for v in e if not region.contains_node(v)]
        
        self.set_goal = SetGoal(nodes)
        self.final_goal = LocationGoal(final_goal)

    def heuristic(self, p: Tuple[int, int]) -> float:
        return self.set_goal.heuristic(p)
        
    def satisfied(self, p: Tuple[int, int]) -> bool:
        return self.set_goal.satisfied(p)

@dataclass
class ConstraintSet:
    constraints: Dict = None
    
    def __post_init__(self):
        self.constraints = self.constraints or {}
        
    def add(self, constraint: Any):
        self.constraints[constraint] = True
        
    def __contains__(self, item: Any) -> bool:
        if isinstance(item, PathEdge):
            return (item in self.constraints or 
                   item.compliment() in self.constraints)
        return item in self.constraints