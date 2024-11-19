# core.py
from dataclasses import dataclass
import networkx as nx
from typing import Dict, List, Tuple, Any
import copy

from .mapf_utils import (
    Environment,
    GridWorld,
    PathVertex,
    PathEdge,
    Path,
    Goal,
    LocationGoal,
    MAPFSolution
)


class GridRegion:
    def __init__(self, grid_world: GridWorld, location: Tuple[int, int], size: Tuple[int, int]):
        self.size = size
        self.location = location
        self.boundary = []
        nodes = []
        for node in grid_world.G.nodes:
            if location[0] <= node[0] < location[0]+size[0]:
                if location[1] <= node[1] < location[1]+size[1]:
                    nodes.append(node)
                    if location[0] == node[0]:
                        self.boundary.append(node)
                    elif location[0]+size[0]-1 == node[0]:
                        self.boundary.append(node)
                    elif location[1] == node[1]:
                        self.boundary.append(node)
                    elif location[1]+size[1]-1 == node[1]:
                        self.boundary.append(node)

        self.G = nx.subgraph(grid_world.G, nodes)

    def contains_node(self, u: tuple) -> bool:
        return u in self.G.nodes
    
    def contains_edge(self, u: tuple, v: tuple) -> bool:
        return (u,v) in self.G.edges

@dataclass
class CBSNode:
    x: Dict[int, PathVertex]
    goals: Dict[int, Goal]

    def __init__(self, x: Dict[int, PathVertex], goals: Dict[int, Goal]):
        self.x = x
        self.goals = goals
        self.constraints = {}
        self.l = {}
        self.paths = {}
        self.vertexes = {}
        self.edges = {}
        self.conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.lower_bound = 0

    def apply_constraint(self, id: int, c: Any) -> None:
        if id in self.constraints:
            self.constraints[id][c] = True
        else:
            self.constraints[id] = {c: True}

    def detect_conflicts(self) -> None:
        vertexes = {}
        edges = {}
        conflicts = []
        for id in self.paths:
            path = self.paths[id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                if v in vertexes:
                    other = vertexes[v]
                    if other[0] != id:
                        conflicts.append([(id,v), other])
                else:
                    vertexes[v] = (id,e)
                if e.compliment() in edges:
                    other = edges[e.compliment()]
                    if other[0] != id:
                        conflicts.append([(id,e), other])
                else:
                    edges[e] = (id,e)
        self.conflicts = conflicts
        self.conflict_count = len(conflicts)
    
    def compute_cost(self) -> None:
        self.cost = sum(len(self.paths[id]) for id in self.paths)
        self.lower_bound = sum(self.l[id] for id in self.l)
    
    def branch(self, id: int, c: Any) -> 'CBSNode':
        node = copy.deepcopy(self)
        node.apply_constraint(id, c)
        node.paths = copy.deepcopy(self.paths)
        node.l = copy.deepcopy(self.l)
        return node

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, CBSNode):
            raise ValueError(f'Cannot compare CBSNode to other of type {type(other)}')
        return self.cost < other.cost

class HierarchicalEnvironment:
    def __init__(self, gridworld: GridWorld, region_graph: nx.Graph):
        self.gridworld = gridworld
        self.region_graph = region_graph
        self.action_generators = {}  

    def contains_node(self, u: tuple) -> bool:
        return self.gridworld.contains_node(u)
    
    def contains_edge(self, u: tuple, v: tuple) -> bool:
        return self.gridworld.contains_edge(u,v)
    
    def dense_matrix(self):
        return self.gridworld.dense_matrix()

class HCBSNode:
    def __init__(self, x: Dict[int, PathVertex], final_goals: Dict[int, Any], region_paths: Dict[int, List[Any]]):
        self.x = x
        self.final_goals = final_goals
        self.region_paths = region_paths
        self.partial_paths = {}
        self.trip_idx = dict((id, 0) for id in final_goals)
        self.region_conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.goal_cost = 0
        self.lb = {}
        self.lower_bound = 0
        self.cbs_nodes = {}
        self.constraints = {}
        
    def detect_conflicts(self) -> None:
        conflict_count = 0
        conflicts = []
        vertexes = {}
        edges = {}
        for r in self.cbs_nodes:
            N = self.cbs_nodes[r]
            for id in N.paths:
                path = N.paths[id]
                for i in range(len(path)):
                    u = path[i]
                    if u in vertexes:
                        other = vertexes[u]
                        if other[1] != id:
                            conflicts.append([(r,id,u), other])
                            conflict_count += 1
                    else:
                        vertexes[u] = (r,id,u)
                    
                    if i < len(path)-1:
                        v = path[i+1]
                        e = PathEdge(u.pos, v.pos, u.t)
                        if e.compliment() in edges:
                            other = edges[e.compliment()]
                            if other[1] != id:
                                conflicts.append([(r,id,e), other])
                                conflict_count += 1
                        else:
                            edges[e] = (r,id,e)

        self.region_conflicts = conflicts
        self.conflict_count = conflict_count

    def compute_cost(self) -> None:
        self.goal_cost = sum(-self.trip_idx[id] for id in self.trip_idx)
        self.cost = 0
        self.lower_bound = 0

        for id in self.region_paths:
            region_path = self.region_paths[id]
            trip_idx = self.trip_idx[id]
            for r in region_path[0:trip_idx]:
                path_cost = len(self.partial_paths[r][id])
                self.lower_bound += path_cost
                self.cost += path_cost
        for r in self.cbs_nodes:
            N = self.cbs_nodes[r]
            self.cost += N.cost
        for r in self.lb:
            self.lower_bound += self.lb[r]

    def make_solution(self) -> MAPFSolution:
        paths = {}
        for id in self.region_paths:
            region_path = self.region_paths[id]
            for r in region_path:
                path = copy.copy(self.partial_paths[r][id])
                if id not in paths:
                    paths[id] = path
                else:
                    paths[id] += path
        return MAPFSolution(paths)

    def __lt__(self, other: Any) -> bool:
        return False  # Default comparison for priority queue
    
