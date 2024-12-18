#search.py
from heapq import heappush, heappop, heapify
import time
import copy
from typing import Dict, List, Tuple, Optional, Any

from .mapf_utils import (
    PathVertex,
    PathEdge,
    Path,
    Goal,
    LocationGoal
)

from .core import (
    CBSNode,
    HCBSNode,
    HierarchicalEnvironment
)
from .constraints import BoundaryGoal, RegionActionGenerator

def focal_astar(
    action_gen: RegionActionGenerator,
    V: Dict,
    E: Dict,
    v: PathVertex,
    goal: Goal,
    constraints: Dict,
    omega: float
) -> Tuple[Optional[Path], float, float]:
    
    OPEN = []
    open_finder = {}
    FOCAL = []
    focal_finder = {}
    predecessors = {}
    h = lambda loc: goal.heuristic(loc)
    d = {}
    g = {}
    f = {}

    if v in constraints:
        return None, float('inf'), float('inf')
    
    predecessors[v] = None
    d_score = 0
    if v in V:
        d_score += 1
    g_score = 1
    f_score = g_score + h(v.pos)
    d[v] = d_score
    g[v] = g_score
    f[v] = f_score 

    entry = [f_score, v]
    open_finder[v] = entry
    f_best = f_score
    heappush(OPEN, entry)

    entry = [d_score, f_score, v]
    focal_finder[v] = entry
    heappush(FOCAL, entry)

    while len(FOCAL) > 0 or len(OPEN) > 0:
        if OPEN[0][0] != f_best:
            f_best = OPEN[0][0]
            OLD_FOCAL = FOCAL
            FOCAL = []
            while len(OLD_FOCAL) > 0:
                entry = heappop(OLD_FOCAL)
                v = entry[-1]
                focal_finder.pop(v)
                if f_score <= omega*f_best:
                    focal_finder[v] = entry
                    heappush(FOCAL, [d_score, f_score, v])
        
        if len(FOCAL) > 0:
            entry = FOCAL[0]
            d_score = entry[0]
            f_score = entry[1]
            v = entry[2]
            if goal.satisfied(v.pos):
                vertexes = []
                while predecessors[v] is not None:
                    vertexes.append(v)
                    v = predecessors[v]
                vertexes.append(v)
                vertexes.reverse()
                return Path(vertexes), len(vertexes), OPEN[0][0]
            heappop(FOCAL)
            focal_finder.pop(v)
        else:
            entry = OPEN[0]
            f_score = entry[0]
            v = entry[1]
            if goal.satisfied(v.pos):
                vertexes = []
                while predecessors[v] is not None:
                    vertexes.append(v)
                    v = predecessors[v]
                vertexes.append(v)
                vertexes.reverse()
                return Path(vertexes), len(vertexes), OPEN[0][0]
            heappop(OPEN)
            open_finder.pop(v)

        new_nodes = []
        for (u,e) in action_gen.actions(v):
            if u in constraints or e in constraints or e.compliment() in constraints:
                continue
            new_nodes.append(u)

        for u in new_nodes:
            if u in g:
                if g[v] + 1 < g[u]:
                    predecessors[u] = v
                    d_score = d[v]
                    g_score = g[v] + 1
                    f_score = g_score + h(u.pos)
                    e = PathEdge(v.pos, u.pos, v.t)
                    if v in V:
                        d_score += 1
                    if e in E or e.compliment() in E:
                        d_score += 1
                    d[u] = d_score
                    g[u] = g_score
                    f[u] = f_score
                    if u not in open_finder:
                        open_entry = [f_score, u]
                        open_finder[u] = open_entry
                        heappush(OPEN, open_entry)
                    else:
                        open_entry = open_finder[u]
                        if f_score != open_entry[0]:
                            open_entry[0] = f_score
                            heapify(OPEN)
                    if f_score <= OPEN[0][0]*omega:
                        if u not in focal_finder:
                            focal_entry = [d_score, f_score, u]
                            focal_finder[u] = focal_entry
                            heappush(FOCAL, focal_entry)
                        else:
                            focal_entry = focal_finder[u]
                            if focal_entry[0] != d_score or focal_entry[1] != f_score:
                                focal_entry[0] = d_score
                                focal_entry[1] = f_score
                                heapify(FOCAL)
            else:
                predecessors[u] = v 
                d_score = d[v]
                g_score = g[v] + 1
                f_score = g_score + h(u.pos)
                e = PathEdge(v.pos, u.pos, v.t)
                if v in V:
                    d_score += 1
                if e in E or e.compliment() in E:
                    d_score += 1
                d[u] = d_score
                g[u] = g_score
                f[u] = f_score
                entry = [f_score, u]
                open_finder[u] = entry
                heappush(OPEN, entry)
                if f_score <= OPEN[0][0]*omega:
                    entry = [d_score, f_score, u]
                    focal_finder[u] = entry
                    heappush(FOCAL, entry)

    return None, float('inf'), float('inf')

def a_star_search(
    action_gen: RegionActionGenerator,
    V: Dict,
    E: Dict,
    v: PathVertex,
    goal: Goal,
    constraints: Dict
) -> Tuple[Optional[Path], float, float]:

    if v in constraints:
        return None, float('inf'), float('inf')
        
    OPEN = []
    came_from = {v: None}
    g_score = {v: 1}
    f_score = g_score[v] + goal.heuristic(v.pos)
    heappush(OPEN, [f_score, v])

    while OPEN:
        f, current = heappop(OPEN)
        if goal.satisfied(current.pos):
            # Build path like focal_astar does
            vertices = []
            while current is not None:
                vertices.append(current)
                current = came_from[current]
            vertices.reverse()
            return Path(vertices), len(vertices), f_score
            
        for (next_v, e) in action_gen.actions(current):
            if next_v in constraints or e in constraints:
                continue
            new_g = g_score[current] + 1
            if next_v not in g_score or new_g < g_score[next_v]:
                came_from[next_v] = current
                g_score[next_v] = new_g 
                f_score = new_g + goal.heuristic(next_v.pos)
                heappush(OPEN, [f_score, next_v])

    return None, float('inf'), float('inf')

def update_paths(node: HCBSNode, ids: List[int], action_gen: RegionActionGenerator, search_type: str, omega: float = 1.0) -> None:
    for id in ids:
        if id in node.vertexes:
            del node.vertexes[id]
        if id in node.edges:
            del node.edges[id]

    V = {v: True for id in node.vertexes for v in node.vertexes[id]}
    E = {e: True for id in node.edges for e in node.edges[id]}
    
    for id in ids:
        goal = node.goals[id]
        c = node.constraints.get(id, {})
        if search_type == 'focal':
            path, cost, lb = focal_astar(action_gen, V, E, node.x[id], goal, c, omega)
        else:
            path, cost, lb = a_star_search(action_gen, V, E, node.x[id], goal, c)
        node.vertexes[id] = {}
        node.edges[id] = {}
        
        if path is not None:
            node.paths[id] = path
            node.l[id] = lb
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                node.vertexes[id][v] = True
                node.edges[id][e] = True
        else:
            node.paths[id] = Path([node.x[id]])
            node.l[id] = float('inf')
            node.cost = float('inf')
            break

def conflict_based_search(
    root: CBSNode,
    action_gen: RegionActionGenerator,
    search_type: str = 'focal',
    omega: float = 1.0,  # Only used for focal
    maxtime: float = 60.0,
    verbose: bool = False
) -> Tuple[CBSNode, float]:
    
    clock_start = time.time()
    ids = list(root.goals.keys())
    update_paths(root, ids, action_gen, omega)
    root.detect_conflicts()
    root.compute_cost()
    
    O = [[root.lower_bound, root]]
    F = [[root.conflict_count, root]]
    
    while len(O) > 0 or len(F) > 0:
        if time.time() - clock_start > maxtime:
            if verbose:
                print('CBS timeout')
            root.cost = float('inf')
            return root, float('inf')

        node = None
        while len(F) > 0:
            entry = heappop(F)
            if entry[1].cost <= omega*O[0][0]:
                node = entry[1]
                if verbose:
                    print('CBS popped from F')
                break

        if node is None:
            lower_bound, node = heappop(O)
            if verbose:
                print('CBS popped from O')

        if node.conflict_count > 0:
            if verbose:
                print(f'Current conflict count {node.conflict_count}')
            conflicts = node.conflicts[0]
            for (id, c) in conflicts:
                if verbose:
                    print(f'Applying constraint {c} to {id}')
                new_node = node.branch(id, c)
                update_paths(new_node, [id], action_gen, omega)
                new_node.detect_conflicts()
                new_node.compute_cost()
                if new_node.lower_bound < float('inf'):
                    heappush(O, [new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][0]:
                        heappush(F, [new_node.conflict_count, new_node])
        else:
            if verbose:
                print('CBS solution found')
            lb = O[0][0] if len(O) > 0 else node.lower_bound
            return node, lb

    if verbose:
        print('Infeasible CBS problem')
    node.cost = float('inf')
    return node, float('inf')

class PreferredBoundaryGoal(Goal):
    """A goal that prefers certain boundary points but allows any valid boundary crossing"""
    def __init__(self, env: HierarchicalEnvironment, source: tuple, dest: tuple, 
                 final_goal: tuple, preferred_point: tuple = None):
        if (source, dest) not in env.region_graph.edges:
            raise ValueError(f"source {source} and dest {dest} are not connected in the region graph")
        
        # Get all possible boundary points
        edges = env.region_graph.edges[source,dest]['boundary']
        region = env.region_graph.nodes[source]['env']
        self.boundary_points = [v for e in edges for v in e if not region.contains_node(v)]
        
        # Store preferred point if provided
        self.preferred_point = preferred_point
        self.final_goal = final_goal
        
        # Weight for preferred point in heuristic
        self.preference_weight = 0.9  # Can be tuned
        
    def heuristic(self, pos: tuple) -> float:
        # Calculate distance to all boundary points
        min_boundary_dist = min(abs(pos[0] - b[0]) + abs(pos[1] - b[1]) 
                              for b in self.boundary_points)
        
        if self.preferred_point is not None:
            # Calculate distance to preferred point
            preferred_dist = (abs(pos[0] - self.preferred_point[0]) + 
                            abs(pos[1] - self.preferred_point[1]))
            
            # Weighted combination of distances
            return (self.preference_weight * preferred_dist + 
                   (1 - self.preference_weight) * min_boundary_dist)
        
        return min_boundary_dist
    
    def satisfied(self, pos: tuple) -> bool:
        return pos in self.boundary_points

def init_hcbs(env: HierarchicalEnvironment, 
              x: Dict, 
              final_goals: Dict, 
              region_paths: Dict,
              region_boundary_points: Dict) -> HCBSNode:
    
    root = HCBSNode(x, final_goals, region_paths)
    cbs_nodes = {r: CBSNode({},{}) for r in env.region_graph.nodes}
    root.cbs_nodes = cbs_nodes
    
    for r in cbs_nodes:
        N = cbs_nodes[r]
        renv = env.region_graph.nodes[r]['env']
        agents = [id for id in x if renv.contains_node(x[id].pos)]
        
        for id in agents:
            N.x[id] = x[id]
            region_path = region_paths[id]
            
            if len(region_path) == 1:
                # Single region case - goal is final destination
                N.goals[id] = LocationGoal(final_goals[id])
            else:
                # Get the next region
                r2 = region_path[1]
                
                # Check if we have a preferred boundary point
                preferred_point = None
                if (id in region_boundary_points and 
                    r in region_boundary_points[id] and 
                    'exit' in region_boundary_points[id][r]):
                    preferred_point = region_boundary_points[id][r]['exit']
                
                # Create goal with preferred point
                N.goals[id] = PreferredBoundaryGoal(
                    env, r, r2, final_goals[id], preferred_point
                )
    
    # Initialize action generators
    for r in env.region_graph.nodes:
        region = env.region_graph.nodes[r]['env']
        env.action_generators[r] = RegionActionGenerator(env.gridworld, region)
        
    return root

def hierarchical_cbs(
    root: HCBSNode,
    env: HierarchicalEnvironment,
    region_boundary_points: Dict,  
    search_type: str = 'focal',
    omega: float = 1.0,  
    maxtime: float = 60.0,
    cbs_maxtime: float = 30.0,
    verbose: bool = False
) -> Optional[HCBSNode]:
    
    clock_start = time.time()

    for r in root.cbs_nodes:
        update_region(root, env, r, omega, cbs_maxtime)
    root.compute_cost()
    root.detect_conflicts()

    O = [[root.goal_cost, root.lower_bound, root]]
    F = [[root.goal_cost, root.conflict_count, root]]
    
    while len(O) > 0 or len(F) > 0:
        if time.time() - clock_start > maxtime:
            if verbose:
                print('HCBS timeout')
            return None

        M = None
        while len(F) > 0:
            goal_cost, conflict_count, M = heappop(F)
            if M.cost <= omega * O[0][1]:
                break
            M = None
            
        if M is None:
            gc, f, M = heappop(O)

        if M.conflict_count > 0:
            conflict = M.region_conflicts[0]
            for (r, id, c) in conflict:
                if verbose:
                    print(f'Branching at region {r} with constraint {c} applied to agent {id}')
                new_node = branch_hcbs(M, r, id, c)
                update_region(new_node, env, r, omega, cbs_maxtime)
                new_node.compute_cost()
                new_node.detect_conflicts()
                
                if new_node.lower_bound < float('inf'):
                    heappush(O, [new_node.goal_cost, new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][1]:
                        heappush(F, [new_node.goal_cost, new_node.conflict_count, new_node])
        else:
            if all(M.trip_idx[id] == len(M.region_paths[id]) for id in M.trip_idx):
                if verbose:
                    print('HCBS successful')
                return M
            else:
                if verbose:
                    print(f'# of completed trips {-M.goal_cost}')
                    print('advancing agents...')
                new_node = advance_agents(M, env, region_boundary_points)  # Pass boundary points
                for r in new_node.cbs_nodes:
                    update_region(new_node, env, r, omega, cbs_maxtime)
                
                new_node.compute_cost()
                new_node.detect_conflicts()
                if new_node.lower_bound < float('inf'):
                    heappush(O, [new_node.goal_cost, new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][1]:
                        heappush(F, [new_node.goal_cost, new_node.conflict_count, new_node])
    
    return None

def update_region(M: HCBSNode, env: HierarchicalEnvironment, r: tuple, omega: float, cbs_maxtime: float) -> None:
    constraints = M.constraints.get(r, {})
    M.constraints[r] = constraints
    action_gen = RegionActionGenerator(env.gridworld, env.region_graph.nodes[r]['env'], constraints=constraints)
    
    N, lb = conflict_based_search(M.cbs_nodes[r], action_gen, omega, maxtime=cbs_maxtime)
    M.lb[r] = lb
    M.cbs_nodes[r] = N
    paths = copy.deepcopy(N.paths)
    
    if r in M.partial_paths:
        M.partial_paths[r].update(paths)
    else:
        M.partial_paths[r] = paths

def branch_hcbs(node: HCBSNode, r: tuple, id: int, c: Any) -> HCBSNode:
    M = copy.deepcopy(node)
    N = M.cbs_nodes[r]
    N.apply_constraint(id, c)
    return M

def advance_agents(node: HCBSNode, env: HierarchicalEnvironment, region_boundary_points: Dict) -> HCBSNode:
    M = copy.deepcopy(node)
    
    # Advance trip indices
    for id in M.trip_idx:
        region_path = M.region_paths[id]
        if M.trip_idx[id] < len(region_path):
            M.trip_idx[id] += 1

    # Create new CBS nodes
    M.cbs_nodes = {r: CBSNode({},{}) for r in env.region_graph.nodes}
    M.lb = {}
    
    for id in M.trip_idx:
        trip_idx = M.trip_idx[id]
        region_path = M.region_paths[id]
        
        if trip_idx == len(region_path):
            continue
        
        # Apply path constraints to last region
        r1 = region_path[trip_idx-1]
        path = M.partial_paths[r1][id]
        
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t)
            if r1 not in M.constraints:
                M.constraints[r1] = {}
            M.constraints[r1][v] = True
            M.constraints[r1][e] = True
            M.constraints[r1][e.compliment()] = True
            
        if trip_idx == len(region_path):
            continue
        
        # Set starting position in next region
        r2 = region_path[trip_idx]
        M.cbs_nodes[r2].x[id] = path[-1]
        
        # Set goal for next region
        if r2 == region_path[-1]:
            M.cbs_nodes[r2].goals[id] = LocationGoal(M.final_goals[id])
        else:
            r3 = region_path[trip_idx+1]
            
            # Get preferred boundary point if available
            preferred_point = None
            if (id in region_boundary_points and 
                r2 in region_boundary_points[id] and 
                'exit' in region_boundary_points[id][r2]):
                preferred_point = region_boundary_points[id][r2]['exit']
            
            # Create goal with preferred point
            M.cbs_nodes[r2].goals[id] = PreferredBoundaryGoal(
                env, r2, r3, M.final_goals[id], preferred_point
            )
            
    return M