import numpy as np
import torch
from queue import PriorityQueue
import heapq
from typing import Dict, List, Tuple

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_canonical_edge(u, v):
    """Return edge tuple in a consistent order"""
    return tuple(sorted([u, v]))


def dijkstra_shortest_path(dense_matrix: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Simplified Dijkstra for grid with uniform costs
    Args:
        dense_matrix: 2D grid where 0 is free space and 1 is obstacle
        start: Starting (row, col) position
        end: Goal (row, col) position
    Returns:
        List of (row, col) positions for shortest path
    """
    grid_size = len(dense_matrix)
    distances = {}
    previous = {}
    pq = []
    visited = set()
    
    # Initialize distances
    for i in range(grid_size):
        for j in range(grid_size):
            distances[(i, j)] = float('inf')
    distances[start] = 0
    heapq.heappush(pq, (0, start))
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current == end:
            break
            
        if current in visited:
            continue
            
        visited.add(current)
        
        # Check 4-connected neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_i, next_j = current[0] + di, current[1] + dj
            
            if (0 <= next_i < grid_size and 
                0 <= next_j < grid_size and 
                dense_matrix[next_i][next_j] == 0):
                
                neighbor = (next_i, next_j)
                if neighbor in visited:
                    continue
                    
                distance = current_distance + 1
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
    
    return path[::-1]

def manhattan_distance(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])