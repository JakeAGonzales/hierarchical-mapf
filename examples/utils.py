from typing import Tuple, Set, List
import os

def read_map_file(filename: str) -> Tuple[int, int, Set[Tuple[int, int]]]:
    """Read a .map file and return height, width, and obstacles."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    obstacles = set()
    for i, line in enumerate(lines[4:]):
        for j, char in enumerate(line.strip()):
            if char == '@':
                obstacles.add((i, j))
    
    return height, width, obstacles

def get_boundary_positions(height: int, width: int, 
                         obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Get valid positions on the boundary of the map."""
    boundary = []
    for i in range(height):
        if (i, 0) not in obstacles:
            boundary.append((i, 0))
        if (i, width-1) not in obstacles:
            boundary.append((i, width-1))
    for j in range(1, width-1):
        if (0, j) not in obstacles:
            boundary.append((0, j))
        if (height-1, j) not in obstacles:
            boundary.append((height-1, j))
    return boundary

def ensure_dir(directory: str):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)