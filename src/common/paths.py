from typing import List, Tuple, Any
import copy

class PathVertex:
    def __init__(self, pos: Tuple[int, int], time: int):
        self.pos = pos
        self.t = time
        
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PathVertex) and self.pos == other.pos and self.t == other.t
        
    def __hash__(self) -> int:
        return hash((self.pos, self.t))
        
    def __str__(self) -> str:
        return f"({self.pos}, t={self.t})"

class PathEdge:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], time: int):
        self.start = start
        self.end = end
        self.time = time
        
    def reverse(self) -> 'PathEdge':
        return PathEdge(self.end, self.start, self.time)
        
    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, PathEdge) and 
                self.time == other.time and 
                {self.start, self.end} == {other.start, other.end})
                
    def __hash__(self) -> int:
        return hash((frozenset([self.start, self.end]), self.time))


class Path:
    def __init__(self, vertices: List[PathVertex], cost: float = None):
        self.vertices = vertices 
        self.cost = cost if cost is not None else len(vertices) - 1

    def __getitem__(self, i):
        return self.vertices[i]

    def __len__(self):
        return len(self.vertices)
        
    def __getitem__(self, idx: int) -> PathVertex:
        return self.vertices[idx]
        
    def get_cost(self) -> int:
        return len(self.vertices) - 1
        
    def get_path_coordinates(self) -> List[Tuple[int, int]]:
        return [v.pos for v in self.vertices]
        
    def validate_continuity(self) -> bool:
        if len(self.vertices) < 2:
            return True
            
        for i in range(len(self.vertices) - 1):
            curr, next = self.vertices[i], self.vertices[i+1]
            if next.t != curr.t + 1:
                return False
        return True