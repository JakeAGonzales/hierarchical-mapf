from typing import Dict, Tuple, Any
from collections import defaultdict

from ..common import get_canonical_edge, dijkstra_shortest_path

def frank_wolfe_step(game: Any) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """Calculate new flows using the Frank-Wolfe algorithm step"""
    new_flows = defaultdict(float)
    for od_pair in game.od_pairs:
        start, end = od_pair
        path = dijkstra_shortest_path(game.graph, start, end, game.edge_costs)
        if path:
            demand = game.demands[od_pair]
            for i in range(len(path) - 1):
                edge = get_canonical_edge(path[i], path[i+1])
                new_flows[edge] += demand
    return new_flows

def normalize_flows(flows: Dict) -> Dict:
    """Normalize flow values to be between 0 and 1"""
    max_flow = max(flows.values())
    if max_flow > 0:
        return {edge: flow / max_flow for edge, flow in flows.items()}
    return flows

def calculate_edge_costs(flows: Dict) -> Dict:
    """Calculate edge costs based on current flows"""
    return {edge: 1 + 0.5 * flow for edge, flow in flows.items()}