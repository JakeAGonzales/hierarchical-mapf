import copy
from typing import Tuple, Dict, Any

class PathVertex:
    def __init__(self, pos: Tuple[int, int], time: int):
        self.pos = pos
        self.t = time

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PathVertex) and self.pos == other.pos and self.t == other.t

    def __hash__(self) -> int:
        return hash((self.pos, self.t))

    def __lt__(self, other: Any) -> bool:
        return (self.t, self.pos) < (other.t, other.pos)

class PathEdge:
    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], t: int):
        self.p1 = p1
        self.p2 = p2
        self.t = t

    def compliment(self) -> 'PathEdge':
        return PathEdge(self.p2, self.p1, self.t)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PathEdge):
            return False
        return self.t == other.t and {self.p1, self.p2} == {other.p1, other.p2}

    def __hash__(self) -> int:
        return hash((frozenset([self.p1, self.p2]), self.t))

class ConstraintSet:
    def __init__(self, data: list = None):
        self._hashmap: Dict = {}
        if data:
            for x in data:
                self._hashmap[x] = True

    def insert(self, x: Any):
        self._hashmap[x] = True

    def __contains__(self, x: Any) -> bool:
        if isinstance(x, PathVertex):
            return x in self._hashmap
        elif isinstance(x, PathEdge):
            return x in self._hashmap or x.compliment() in self._hashmap
        return False