from typing import Callable, List, Dict, Set, Tuple, Optional, Any
import heapq
from .environment import Environment
from .graph import manhattan_distance, get_canonical_edge

def a_star_search(
    env: Environment,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Optional[Callable] = None
) -> List[Tuple[int, int]]:
    if heuristic is None:
        heuristic = manhattan_distance
        
    queue = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while queue:
        _, current = heapq.heappop(queue)
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
            
        for next_pos in env.get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                heapq.heappush(queue, (priority, next_pos))
                came_from[next_pos] = current
                
    return []  # No path found

def dijkstra_shortest_path(
    graph: Any,  # Can be NetworkX graph or Environment
    start: Tuple[int, int],
    end: Tuple[int, int],
    edge_costs: Optional[Dict] = None,
    get_neighbors: Optional[Callable] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Efficient Dijkstra implementation for finding shortest path with custom edge costs.
    
    Args:
        graph: Graph structure (NetworkX graph or Environment)
        start: Starting node coordinates
        end: End node coordinates
        edge_costs: Dictionary of edge costs with canonical edge tuples as keys
                   If None, assumes uniform cost of 1
        get_neighbors: Optional function to get neighbors of a node
                      If None, uses graph.neighbors or graph.get_neighbors
    
    Returns:
        List of coordinates representing shortest path, or None if no path exists
    """
    # Set up neighbor function based on graph type
    if get_neighbors is None:
        if hasattr(graph, 'get_neighbors'):
            get_neighbors = graph.get_neighbors
        elif hasattr(graph, 'neighbors'):
            get_neighbors = graph.neighbors
        else:
            raise ValueError("Graph must have neighbors or get_neighbors method")

    # Initialize data structures
    distances = {}
    previous = {}
    pq = [(0, start)]
    visited = set()
    
    # Initialize distances to infinity
    for node in [start, end]:
        distances[node] = float('inf')
    distances[start] = 0
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current == end:
            break
            
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check all neighbors
        for neighbor in get_neighbors(current):
            if neighbor in visited:
                continue
                
            # Calculate edge cost
            if edge_costs is not None:
                edge = get_canonical_edge(current, neighbor)
                edge_cost = edge_costs[edge]
            else:
                edge_cost = 1
                
            distance = current_distance + edge_cost
            
            # If we found a better path, update it
            if distance < distances.get(neighbor, float('inf')):
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    # If we didn't reach the end, no path exists
    if distances.get(end, float('inf')) == float('inf'):
        return None
        
    # Reconstruct the path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
        
    return path[::-1]

def dijkstra_all_pairs(
    graph: Any,
    nodes: List[Tuple[int, int]],
    edge_costs: Optional[Dict] = None,
    get_neighbors: Optional[Callable] = None
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]:
    """
    Calculate shortest path distances between all pairs of given nodes.
    
    Args:
        graph: Graph structure (NetworkX graph or Environment)
        nodes: List of node coordinates to calculate distances between
        edge_costs: Dictionary of edge costs with canonical edge tuples as keys
                   If None, assumes uniform cost of 1
        get_neighbors: Optional function to get neighbors of a node
    
    Returns:
        Dictionary mapping (source, target) tuples to shortest path distances
    """
    distances = {}
    
    for source in nodes:
        # Run Dijkstra's from source
        pq = [(0, source)]
        dist = {source: 0}
        visited = set()
        
        while pq:
            d, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Record distance if current node is in our target set
            if current in nodes:
                distances[(source, current)] = d
                
            # Explore neighbors
            for neighbor in get_neighbors(current):
                if neighbor in visited:
                    continue
                    
                if edge_costs is not None:
                    edge = get_canonical_edge(current, neighbor)
                    cost = edge_costs[edge]
                else:
                    cost = 1
                    
                new_dist = d + cost
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return distances

def get_path_cost(
    path: List[Tuple[int, int]],
    edge_costs: Optional[Dict] = None
) -> float:
    """
    Calculate the total cost of a path using given edge costs.
    
    Args:
        path: List of coordinates representing the path
        edge_costs: Dictionary of edge costs with canonical edge tuples as keys
                   If None, assumes uniform cost of 1
    
    Returns:
        Total path cost
    """
    if not path or len(path) < 2:
        return 0
        
    total_cost = 0
    for i in range(len(path) - 1):
        if edge_costs is not None:
            edge = get_canonical_edge(path[i], path[i + 1])
            total_cost += edge_costs[edge]
        else:
            total_cost += 1
            
    return total_cost