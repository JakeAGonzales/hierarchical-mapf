from typing import List, Tuple, Dict, Any
import random

def get_grid_boundary_positions(game: Any) -> List[Tuple[int, int]]:
    """Get all boundary positions of the full grid"""
    boundary = []
    
    # Top and bottom edges
    for j in range(game.grid_size):
        boundary.append((0, j))
        boundary.append((game.grid_size - 1, j))
        
    # Left and right edges (excluding corners)
    for i in range(1, game.grid_size - 1):
        boundary.append((i, 0))
        boundary.append((i, game.grid_size - 1))
        
    return boundary

def get_subregion_boundary_positions(game: Any, region_i: int, region_j: int) -> List[Tuple[int, int]]:
    """Get boundary positions for a subregion"""
    start_i = region_i * game.subregion_size
    start_j = region_j * game.subregion_size
    end_i = start_i + game.subregion_size
    end_j = start_j + game.subregion_size
    
    boundary_positions = []
    
    # Top and bottom edges
    for j in range(start_j, end_j):
        boundary_positions.append((start_i, j))
        boundary_positions.append((end_i - 1, j))
        
    # Left and right edges (excluding corners)
    for i in range(start_i + 1, end_i - 1):
        boundary_positions.append((i, start_j))
        boundary_positions.append((i, end_j - 1))
        
    return boundary_positions

def get_boundary_positions(game: Any) -> List[Tuple[int, int]]:
    """Get all boundary nodes based on selected boundary type"""
    if game.boundary_type == "full_grid":
        return get_grid_boundary_positions(game)
    else:  # all_boundaries
        nodes = []
        for i in range(game.num_subregions):
            for j in range(game.num_subregions):
                nodes.extend(get_subregion_boundary_positions(game, i, j))
        return list(set(nodes))  # Remove duplicates

def generate_od_pairs(game: Any) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], Dict]:
    """Generate origin-destination pairs and their demands"""
    available_positions = get_boundary_positions(game)
    n_starts = min(game.num_od_pairs, len(available_positions) // 2)
    
    # Create copy of positions for destinations
    remaining_positions = available_positions.copy()
    
    # Select unique start positions
    random.shuffle(remaining_positions)
    origins = remaining_positions[:n_starts]
    
    # Remove selected start positions
    for pos in origins:
        remaining_positions.remove(pos)
    
    # Select unique end positions
    random.shuffle(remaining_positions)
    destinations = remaining_positions[:n_starts]
    
    od_pairs = [(start, end) for start, end in zip(origins, destinations)]
    demands = {pair: game.total_flow for pair in od_pairs}
    
    return od_pairs, demands