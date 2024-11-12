import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

from .mapf_utils import PathVertex, LocationGoal

from .core import HierarchicalEnvironment

def column_lattice_obstacles(h: int, w: int, dy: int, dx: int, 
                           obstacle_rows: int, obstacle_cols: int) -> List[Tuple[int, int]]:
    obstacles = []
    for i in range(obstacle_rows):
        for j in range(obstacle_cols):
            row = i*(w+2*dx)+dx
            col = j*(h+2*dy)+dy
            obstacles += [(row+k, col+l) for k in range(h) for l in range(w)]
    return obstacles

def random_problem(
    N_agents: int, 
    env: HierarchicalEnvironment, 
    path_cutoff: int = 10, 
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[int, PathVertex], Dict[int, Tuple[int, int]], Dict[int, List[Tuple[int, int]]]]:
    
    if rng is None:
        rng = np.random.default_rng()

    # Assign start locations to agents
    start_regions = {}
    start_pos = {}
    nodes = list(env.region_graph.nodes)
    
    for id in range(N_agents):
        start_regions[id] = R = nodes[rng.choice(len(nodes))]
        sub_env = env.region_graph.nodes[R]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in start_pos.values() and 
                all(sub_env.contains_node(u) 
                    for u in env.gridworld.G.adj[p])]
        start_pos[id] = locs[rng.choice(len(locs))]

    # Assign random final goal regions
    final_goal_regions = {}
    final_goals = {}
    shortest_path_lens = dict(nx.shortest_path_length(env.region_graph))
    
    for id in start_regions:
        R1 = start_regions[id]
        choices = [R2 for R2 in shortest_path_lens[R1] if shortest_path_lens[R1][R2] < path_cutoff]
        final_goal_regions[id] = R2 = choices[rng.choice(len(choices))]
        sub_env = env.region_graph.nodes[R2]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in final_goals.values() and
                all(sub_env.contains_node(u)
                    for u in env.gridworld.G.adj[p])]
        final_goals[id] = locs[rng.choice(len(locs))]

    # Assemble trip graph with 1-weight edges initially
    trip_graph = nx.Graph()
    for v1 in env.region_graph.nodes:
        edges = []
        for v2 in env.region_graph.adj[v1]:
            edges.append((v1,v2,10))
        trip_graph.add_weighted_edges_from(edges, weight='c')

    # Generate regional paths for agents
    region_paths = {}
    for id in start_regions:
        R1 = start_regions[id]
        R2 = final_goal_regions[id]
        if R1 == R2:
            region_paths[id] = [R1]
            continue
        region_paths[id] = path = [R for R in nx.shortest_path(trip_graph, R1, R2, weight='c')]
        for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = (u,v)
                trip_graph.edges[e]['c'] += 1
                
    x = {id : PathVertex(start_pos[id], 0) for id in start_pos}
    return x, final_goals, region_paths

def get_solution_metrics(solution: Dict[int, List[PathVertex]]) -> Dict[str, float]:
    """
    Calculate various metrics for a solution including makespan, total path length, etc.
    """
    metrics = {}
    
    # Calculate makespan (maximum path length)
    metrics['makespan'] = max(len(path) for path in solution.values())
    
    # Calculate sum of path lengths
    metrics['total_path_length'] = sum(len(path) for path in solution.values())
    
    # Calculate average path length
    metrics['avg_path_length'] = metrics['total_path_length'] / len(solution)
    
    return metrics

def get_path_coordinates(path: List[PathVertex]) -> List[Tuple[int, int]]:
    """
    Extract just the coordinates from a path for visualization or analysis
    """
    return [vertex.pos for vertex in path]

def verify_solution(solution: Dict[int, List[PathVertex]], env: HierarchicalEnvironment) -> bool:
    """
    Verify that a solution is valid (no collisions, follows constraints, etc.)
    """
    # Check that all positions are valid
    for agent_id, path in solution.items():
        for vertex in path:
            if not env.contains_node(vertex.pos):
                return False
            
        # Check that consecutive positions are adjacent
        for i in range(len(path)-1):
            pos1, pos2 = path[i].pos, path[i+1].pos
            if not env.contains_edge(pos1, pos2):
                return False
    
    # Check for vertex conflicts
    time_pos = {}
    for agent_id, path in solution.items():
        for vertex in path:
            key = (vertex.t, vertex.pos)
            if key in time_pos and time_pos[key] != agent_id:
                return False
            time_pos[key] = agent_id
            
    # Check for edge conflicts
    for agent1_id, path1 in solution.items():
        for agent2_id, path2 in solution.items():
            if agent1_id >= agent2_id:
                continue
            for i in range(len(path1)-1):
                for j in range(len(path2)-1):
                    if path1[i].t == path2[j].t:
                        if (path1[i].pos == path2[j+1].pos and 
                            path1[i+1].pos == path2[j].pos):
                            return False
                            
    return True