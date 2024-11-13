# paths.py

from typing import Dict, List, Tuple, Any
from ..common import get_canonical_edge, dijkstra_shortest_path
from .core import AbstractedRoutingGame

def extract_deterministic_paths(game: AbstractedRoutingGame, equilibrium_flows: Dict) -> Tuple[Dict, Dict]:
    """
    Extract paths that follow equilibrium flow patterns, favoring edges with higher flow.
    
    Args:
        game: AbstractedRoutingGame instance
        equilibrium_flows: Dictionary of equilibrium flows
        
    Returns:
        Tuple of (paths dictionary, travel times dictionary)
    """
    paths = {}
    travel_times = {}
    
    # Normalize flows to 0-1 scale
    max_flow = max(equilibrium_flows.values())
    normalized_flows = {
        edge: flow/max_flow for edge, flow in equilibrium_flows.items()
    }
    
    for od_pair in game.od_pairs:
        origin, destination = od_pair
        
        # Create edge weights that favor high-flow edges
        edge_weights = {}
        for edge, flow in normalized_flows.items():
            dist = euclidean_distance(edge[0], edge[1])
            edge_weights[edge] = dist * (1 / (flow + 0.1))
            
        # Find path that best follows equilibrium flow pattern
        equilibrium_path = dijkstra_shortest_path(game.graph, origin, destination, edge_weights)
        
        if equilibrium_path:
            paths[od_pair] = equilibrium_path
            # calculate actual travel time using game's cost function
            travel_time = sum(
                game.edge_cost(
                    get_canonical_edge(equilibrium_path[i], equilibrium_path[i+1]),
                    equilibrium_flows[get_canonical_edge(equilibrium_path[i], equilibrium_path[i+1])]
                )
                for i in range(len(equilibrium_path)-1)
            )
            travel_times[od_pair] = travel_time
            
    return paths, travel_times

def analyze_paths(game: AbstractedRoutingGame, paths: Dict) -> Dict:
    """
    Analyze paths to extract useful information including subregion sequences
    
    Args:
        game: AbstractedRoutingGame instance
        paths: Dictionary of paths from extract_deterministic_paths
        
    Returns:
        Dictionary containing path analysis results
    """
    analysis = {}
    
    for od_pair, path in paths.items():
        origin, destination = od_pair
        
        # Get subregion sequence
        subregion_sequence = extract_subregion_sequence(path, game.subregion_size)
        
        path_length = len(path)
        num_subregions = len(subregion_sequence)
        
        analysis[od_pair] = {
            'subregion_sequence': subregion_sequence,
            'path_length': path_length,
            'num_subregions': num_subregions,
            'origin_subregion': get_subregion_coordinates(origin, game.subregion_size),
            'destination_subregion': get_subregion_coordinates(destination, game.subregion_size)
        }
        
    return analysis

def get_subregion_coordinates(point: Tuple[int, int], subregion_size: int) -> Tuple[int, int]:
    """Convert a point to its subregion coordinates"""
    return (point[0] // subregion_size, point[1] // subregion_size)

def extract_subregion_sequence(path: List[Tuple[int, int]], subregion_size: int) -> List[Tuple[int, int]]:
    """Extract the sequence of subregions that a path passes through"""
    if not path:
        return []
        
    # get sequence of subregions
    subregion_sequence = [get_subregion_coordinates(point, subregion_size) for point in path]
    
    # remove consecutive duplicates while preserving order
    unique_sequence = []
    for subregion in subregion_sequence:
        if not unique_sequence or subregion != unique_sequence[-1]:
            unique_sequence.append(subregion)
            
    return unique_sequence

def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Calculate euclidean distance between two points"""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5