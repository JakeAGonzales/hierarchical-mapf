from typing import Dict, List, Tuple, Any
from ..common import get_canonical_edge, dijkstra_shortest_path

def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def get_path_congestion(path: List[Tuple[int, int]], equilibrium_flows: Dict) -> Tuple[float, float]:
    flows = []
    for i in range(len(path) - 1):
        edge = get_canonical_edge(path[i], path[i+1])
        flow = equilibrium_flows.get(edge, 0)
        flows.append(flow)
    
    if flows:
        return sum(flows) / len(flows), max(flows)
    return 0, 0

def find_direct_path(game: Any, origin: Tuple[int, int], destination: Tuple[int, int], 
                    equilibrium_flows: Dict) -> List[Tuple[int, int]]:
    edge_weights = {}
    for edge in game.graph.edges():
        dist = euclidean_distance(edge[0], edge[1])
        edge_weights[get_canonical_edge(*edge)] = dist
    
    return dijkstra_shortest_path(game.graph, origin, destination, edge_weights)

def extract_deterministic_paths(game: Any, equilibrium_flows: Dict) -> Tuple[Dict, Dict]:
    paths = {}
    best_times = {}
    
    max_flow = max(equilibrium_flows.values())
    normalized_flows = {
        edge: flow/max_flow for edge, flow in equilibrium_flows.items()
    }
    
    for od_pair in game.od_pairs:
        origin, destination = od_pair
        
        edge_weights = {}
        for edge, flow in normalized_flows.items():
            dist = euclidean_distance(edge[0], edge[1])
            edge_weights[edge] = dist * (1 / (flow + 0.1))
            
        equilibrium_path = dijkstra_shortest_path(game.graph, origin, destination, edge_weights)
        
        if equilibrium_path:
            paths[od_pair] = equilibrium_path
            travel_time = sum(game.edge_cost(get_canonical_edge(equilibrium_path[i], equilibrium_path[i+1]), 
                                    equilibrium_flows[get_canonical_edge(equilibrium_path[i], equilibrium_path[i+1])])
                     for i in range(len(equilibrium_path)-1))
            best_times[od_pair] = travel_time
            
            path_flows = [normalized_flows[get_canonical_edge(equilibrium_path[i], equilibrium_path[i+1])]
                         for i in range(len(equilibrium_path)-1)]
            
    return paths, best_times

def get_subregion_coordinates(point: Tuple[int, int], subregion_size: int) -> Tuple[int, int]:
    return (point[0] // subregion_size, point[1] // subregion_size)

def extract_subregion_sequence(path: List[Tuple[int, int]], subregion_size: int) -> List[Tuple[int, int]]:
    if not path:
        return []
        
    subregion_sequence = [get_subregion_coordinates(point, subregion_size) for point in path]
    
    unique_sequence = []
    for subregion in subregion_sequence:
        if not unique_sequence or subregion != unique_sequence[-1]:
            unique_sequence.append(subregion)
            
    return unique_sequence

def analyze_paths_subregion_sequences(game: Any, deterministic_paths: Dict) -> Dict:
    path_sequences = {}
    
    for od_pair, path in deterministic_paths.items():
        sequence = extract_subregion_sequence(path, game.subregion_size)
        path_sequences[od_pair] = sequence
        
    return path_sequences