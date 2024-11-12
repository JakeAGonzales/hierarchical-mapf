from typing import Any, Dict, List, Tuple

from mapf_utils import (
    Goal,
    LocationGoal,
    SetGoal,
    PathVertex,
    PathEdge,
    ActionGenerator,
    GridWorld
)

from .core import GridRegion, HierarchicalEnvironment

class BoundaryGoal(Goal):
    def __init__(self, env: HierarchicalEnvironment, source: Tuple[int, int], 
                 dest: Tuple[int, int], final_goal: Tuple[int, int]):
        if (source, dest) not in env.region_graph.edges:
            raise ValueError(f"source {source} and dest {dest} are not connected in the region graph")
        
        edges = env.region_graph.edges[source,dest]['boundary']
        region = env.region_graph.nodes[source]['env']
        nodes = [v for e in edges for v in e if not region.contains_node(v)]
        self.set_goal = SetGoal(nodes)
        self.final_goal = LocationGoal(final_goal)

    def heuristic(self, p: Tuple[int, int]) -> float:
        return self.set_goal.heuristic(p)
    
    def satisfied(self, p: Tuple[int, int]) -> bool:
        return self.set_goal.satisfied(p)

class RegionActionGenerator(ActionGenerator):
    def __init__(self, world: GridWorld, region: GridRegion, constraints: Dict = None):
        self.world = world
        self.region = region
        self.constraints = constraints or {}

    def actions(self, v: PathVertex):
        if self.region.contains_node(v.pos):
            for pos in self.world.G.adj[v.pos]:
                u = PathVertex(pos, v.t+1)
                e = PathEdge(v.pos, pos, v.t)
                if u in self.constraints:
                    continue
                if e in self.constraints:
                    continue
                if e.compliment() in self.constraints:
                    continue
                yield (u,e)