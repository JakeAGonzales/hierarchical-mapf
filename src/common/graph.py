from typing import Tuple
import numpy as np

def get_canonical_edge(u: Tuple[int, int], v: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return tuple(sorted([u, v]))

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_path_distance(path: list) -> float:
    if len(path) < 2:
        return 0
    return sum(manhattan_distance(path[i], path[i+1]) for i in range(len(path)-1))