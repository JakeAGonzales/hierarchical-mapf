from typing import Dict, List, Tuple, Optional, Any
import copy
import heapq
import numpy as np
from time import time

from ..common import PathVertex, PathEdge
from .constraints import LocationGoal, BoundaryGoal, RegionActionGenerator
from .search import focal_search

class CBSNode:
    def __init__(self, x: Dict = None, goals: Dict = None):
        self.x = x or {}  # start vertices for agents
        self.goals = goals or {}  # goals for agents
        self.constraints = {}  # constraints for each agent
        self.l = {}  # path length lower bounds
        self.paths = {}  # paths for each agent
        self.vertexes = {}  # vertices used by each agent
        self.edges = {}  # edges used by each agent
        self.conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.lower_bound = 0

    def apply_constraint(self, agent_id: int, constraint: Any):
        if agent_id in self.constraints:
            self.constraints[agent_id][constraint] = True
        else:
            self.constraints[agent_id] = {constraint: True}

    def detect_conflicts(self):
        vertexes = {}
        edges = {}
        conflicts = []
        
        for agent_id in self.paths:
            path = self.paths[agent_id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                edge = PathEdge(u.pos, v.pos, u.t)
                
                if v in vertexes:
                    other = vertexes[v]
                    if other[0] != agent_id:
                        conflicts.append([(agent_id, v), other])
                else:
                    vertexes[v] = (agent_id, edge)
                    
                if edge.compliment() in edges:
                    other = edges[edge.compliment()]
                    if other[0] != agent_id:
                        conflicts.append([(agent_id, edge), other])
                else:
                    edges[edge] = (agent_id, edge)
                    
        self.conflicts = conflicts
        self.conflict_count = len(conflicts)

    def compute_cost(self):
        self.cost = sum(len(self.paths[agent_id]) for agent_id in self.paths)
        self.lower_bound = sum(self.l[agent_id] for agent_id in self.l)

    def branch(self, agent_id: int, constraint: Any):
        node = copy.deepcopy(self)
        node.apply_constraint(agent_id, constraint)
        node.paths = copy.deepcopy(self.paths)
        node.l = copy.deepcopy(self.l)
        return node

    def __lt__(self, other):
        if not isinstance(other, CBSNode):
            raise ValueError(f'Cannot compare CBSNode to {type(other)}')
        return self.cost < other.cost

def update_paths(node: CBSNode, agent_ids: List[int], action_gen: RegionActionGenerator, omega: float):
    # Clear vertex/edge sets for agents being updated
    for agent_id in agent_ids:
        if agent_id in node.vertexes:
            del node.vertexes[agent_id]
        if agent_id in node.edges:
            del node.edges[agent_id]

    # Create sets of vertices and edges to check during A*
    V = {v: True for agent_id in node.vertexes for v in node.vertexes[agent_id]}
    E = {e: True for agent_id in node.edges for e in node.edges[agent_id]}

    for agent_id in agent_ids:
        node.vertexes[agent_id] = {}
        node.edges[agent_id] = {}
        
        constraints = node.constraints.get(agent_id, {})
        path, cost, lb = focal_search(
            action_gen, V, E, node.x[agent_id], 
            node.goals[agent_id], constraints, omega
        )

        if path is not None:
            node.paths[agent_id] = path
            node.l[agent_id] = lb
            
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                edge = PathEdge(u.pos, v.pos, u.t)
                node.vertexes[agent_id][v] = True
                node.edges[agent_id][edge] = True
        else:
            node.paths[agent_id] = [node.x[agent_id]]
            node.l[agent_id] = np.inf
            node.cost = np.inf
            break

def conflict_based_search(root: CBSNode, action_gen: RegionActionGenerator, 
                         omega: float, maxtime: float = 60.0, verbose: bool = False):
    clock_start = time.time()
    agent_ids = list(root.goals.keys())
    
    update_paths(root, agent_ids, action_gen, omega)
    root.detect_conflicts()
    root.compute_cost()
    
    O = [[root.lower_bound, root]]
    F = [[root.conflict_count, root]]
    
    while O and F:
        if time.time() - clock_start > maxtime:
            if verbose:
                print('CBS timeout')
            root.cost = np.inf
            return root, np.inf

        # Select node from focal or open set
        node = None
        while F:
            entry = heapq.heappop(F)
            if entry[1].cost <= omega * O[0][0]:
                node = entry[1]
                if verbose:
                    print('CBS popped from F')
                break

        if node is None:
            _, node = heapq.heappop(O)
            if verbose:
                print('CBS popped from O')

        if node.conflict_count > 0:
            if verbose:
                print(f'Current conflict count {node.conflict_count}')
            conflicts = node.conflicts[0]
            
            for agent_id, constraint in conflicts:
                if verbose:
                    print(f'Applying constraint {constraint} to agent {agent_id}')
                    
                new_node = node.branch(agent_id, constraint)
                update_paths(new_node, [agent_id], action_gen, omega)
                new_node.detect_conflicts()
                new_node.compute_cost()
                
                if new_node.lower_bound < np.inf:
                    heapq.heappush(O, [new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][0]:
                        heapq.heappush(F, [new_node.conflict_count, new_node])
        else:
            if verbose:
                print('CBS solution found')
            if O:
                lb = O[0][0]
            else:
                lb = node.lower_bound
            return node, lb

    if verbose:
        print('Infeasible CBS problem')
    root.cost = np.inf
    return root, np.inf