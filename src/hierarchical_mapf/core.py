from typing import Dict, List, Tuple, Optional
import copy
import time
import networkx as nx
from dataclasses import dataclass
import heapq

from ..common import PathVertex, PathEdge
from .constraints import LocationGoal, BoundaryGoal
from .search import conflict_based_search, update_paths
from .cbs_node import CBSNode

@dataclass
class HCBSConfig:
    max_time: float = 60.0
    cbs_time: float = 30.0
    omega: float = 1.0
    verbose: bool = False

class HCBSNode:
    def __init__(self, x: Dict = None, final_goals: Dict = None, region_paths: Dict = None):
        self.x = x or {}
        self.final_goals = final_goals or {}
        self.region_paths = region_paths or {}
        self.partial_paths = {}
        self.trip_idx = {id: 0 for id in final_goals} if final_goals else {}
        self.region_conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.goal_cost = 0
        self.lb = {}
        self.lower_bound = 0
        self.cbs_nodes = {}
        self.constraints = {}

    def detect_conflicts(self):
        conflict_count = 0
        conflicts = []
        vertexes = {}
        edges = {}
        
        for r in self.cbs_nodes:
            node = self.cbs_nodes[r]
            for agent_id, path in node.paths.items():
                for i, vertex in enumerate(path):
                    if vertex in vertexes:
                        other = vertexes[vertex]
                        if other[1] != agent_id:
                            conflicts.append([(r, agent_id, vertex), other])
                            conflict_count += 1
                    else:
                        vertexes[vertex] = (r, agent_id, vertex)
                    
                    if i < len(path) - 1:
                        next_vertex = path[i + 1]
                        edge = PathEdge(vertex.pos, next_vertex.pos, vertex.t)
                        if edge.compliment() in edges:
                            other = edges[edge.compliment()]
                            if other[1] != agent_id:
                                conflicts.append([(r, agent_id, edge), other])
                                conflict_count += 1
                        else:
                            edges[edge] = (r, agent_id, edge)

        self.region_conflicts = conflicts
        self.conflict_count = conflict_count

    def compute_cost(self):
        self.goal_cost = sum(-trip for trip in self.trip_idx.values())
        self.cost = 0
        self.lower_bound = 0

        for agent_id, region_path in self.region_paths.items():
            trip_idx = self.trip_idx[agent_id]
            for r in region_path[0:trip_idx]:
                path_cost = len(self.partial_paths[r][agent_id])
                self.lower_bound += path_cost
                self.cost += path_cost
                
        for r, node in self.cbs_nodes.items():
            self.cost += node.cost
            if r in self.lb:
                self.lower_bound += self.lb[r]

    def make_solution(self):
        paths = {}
        for agent_id, region_path in self.region_paths.items():
            agent_path = []
            for r in region_path:
                if r in self.partial_paths and agent_id in self.partial_paths[r]:
                    if not agent_path:
                        agent_path = self.partial_paths[r][agent_id]
                    else:
                        agent_path += self.partial_paths[r][agent_id][1:]
            if agent_path:
                paths[agent_id] = agent_path
        return paths

class HCBS:
    def __init__(self, config: Optional[HCBSConfig] = None):
        self.config = config or HCBSConfig()

    def solve(self, env, x, final_goals, region_paths):
        root = self._init_root(env, x, final_goals, region_paths)
        return self._search(root, env)

    def _init_root(self, env, x, final_goals, region_paths):
        root = HCBSNode(x, final_goals, region_paths)
        root.cbs_nodes = {r: self._create_cbs_node(env, r, x, final_goals, region_paths) 
                         for r in env.region_graph.nodes}
        return root

    def _create_cbs_node(self, env, region, x, final_goals, region_paths):
        node = CBSNode()
        region_env = env.region_graph.nodes[region]['env']
        
        for agent_id, pos in x.items():
            if region_env.contains_node(pos.pos):
                node.x[agent_id] = pos
                path = region_paths[agent_id]
                
                if len(path) == 1:
                    node.goals[agent_id] = LocationGoal(final_goals[agent_id])
                else:
                    next_region = path[1]
                    node.goals[agent_id] = BoundaryGoal(
                        env, region, next_region, final_goals[agent_id]
                    )
        
        return node

    def _search(self, root: HCBSNode, env):
        start_time = time.time()
        
        for r in root.cbs_nodes:
            self._update_region(root, env, r)
            
        root.compute_cost()
        root.detect_conflicts()

        open_set = [(root.goal_cost, root.lower_bound, root)]
        focal_set = [(root.goal_cost, root.conflict_count, root)]

        while open_set and focal_set:
            if time.time() - start_time > self.config.max_time:
                if self.config.verbose:
                    print('HCBS timeout')
                return None

            node = self._select_node(open_set, focal_set)
            
            if node.conflict_count > 0:
                if not self._resolve_conflicts(node, env, open_set, focal_set):
                    continue
            elif self._check_completion(node):
                if self.config.verbose:
                    print('HCBS solution found')
                return node
            else:
                self._advance_agents(node, env, open_set, focal_set)
                
        return None

    def _update_region(self, node, env, region):
        if region in node.constraints:
            constraints = node.constraints[region]
        else:
            constraints = node.constraints[region] = {}
            
        action_gen = env.action_generators[region]
        cbs_node, lb = conflict_based_search(
            node.cbs_nodes[region], 
            action_gen, 
            self.config.omega,
            maxtime=self.config.cbs_time
        )
        
        node.lb[region] = lb
        node.cbs_nodes[region] = cbs_node
        
        paths = copy.deepcopy(cbs_node.paths)
        if region in node.partial_paths:
            node.partial_paths[region].update(paths)
        else:
            node.partial_paths[region] = paths

    def _select_node(self, open_set, focal_set):
        while focal_set:
            entry = heapq.heappop(focal_set)
            if entry[2].cost <= self.config.omega * open_set[0][1]:
                return entry[2]
        return heapq.heappop(open_set)[2]

    def _resolve_conflicts(self, node, env, open_set, focal_set):
        conflict = node.region_conflicts[0]
        success = False
        
        for r, agent_id, constraint in conflict:
            new_node = copy.deepcopy(node)
            new_node.constraints.setdefault(r, {})[constraint] = True
            
            self._update_region(new_node, env, r)
            new_node.compute_cost()
            new_node.detect_conflicts()
            
            if new_node.lower_bound < float('inf'):
                heapq.heappush(open_set, 
                              (new_node.goal_cost, new_node.lower_bound, new_node))
                if new_node.cost <= self.config.omega * open_set[0][1]:
                    heapq.heappush(focal_set, 
                                 (new_node.goal_cost, new_node.conflict_count, new_node))
                success = True
                
        return success

    def _check_completion(self, node):
        return all(node.trip_idx[id] == len(node.region_paths[id]) 
                  for id in node.trip_idx)

    def _advance_agents(self, node, env, open_set, focal_set):
        new_node = self._create_advanced_node(node, env)
        
        for r in new_node.cbs_nodes:
            self._update_region(new_node, env, r)
            
        new_node.compute_cost()
        new_node.detect_conflicts()
        
        if new_node.lower_bound < float('inf'):
            heapq.heappush(open_set, 
                          (new_node.goal_cost, new_node.lower_bound, new_node))
            if new_node.cost <= self.config.omega * open_set[0][1]:
                heapq.heappush(focal_set, 
                             (new_node.goal_cost, new_node.conflict_count, new_node))

    def _create_advanced_node(self, node, env):  # Add env parameter
        new_node = copy.deepcopy(node)
        
        for agent_id in new_node.trip_idx:
            if new_node.trip_idx[agent_id] < len(new_node.region_paths[agent_id]):
                new_node.trip_idx[agent_id] += 1
        
        new_node.cbs_nodes = {r: CBSNode() for r in env.region_graph.nodes}
        new_node.lb = {}
        
        self._setup_advanced_constraints(new_node, env)  # Pass env
        return new_node

    def _setup_advanced_constraints(self, node, env):  # Add env parameter
        for agent_id, trip_idx in node.trip_idx.items():
            region_path = node.region_paths[agent_id]
            if trip_idx == len(region_path):
                continue
                
            prev_region = region_path[trip_idx - 1]
            path = node.partial_paths[prev_region][agent_id]
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = PathEdge(u.pos, v.pos, u.t)
                node.constraints.setdefault(prev_region, {})[v] = True
                node.constraints[prev_region][edge] = True
                node.constraints[prev_region][edge.compliment()] = True
                
            if trip_idx < len(region_path):
                curr_region = region_path[trip_idx]
                node.cbs_nodes[curr_region].x[agent_id] = path[-1]
                
                if curr_region == region_path[-1]:
                    node.cbs_nodes[curr_region].goals[agent_id] = LocationGoal(
                        node.final_goals[agent_id]
                    )
                else:
                    next_region = region_path[trip_idx + 1]
                    node.cbs_nodes[curr_region].goals[agent_id] = BoundaryGoal(
                        env,  # Now env is available
                        curr_region, 
                        next_region, 
                        node.final_goals[agent_id]
                    )