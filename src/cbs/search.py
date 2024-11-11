from typing import Optional, Tuple, Dict
import heapq
from .environment import Environment
from .constraints import PathVertex, PathEdge, ConstraintSet
from ..common.paths import Path

def single_agent_astar(
    env: Environment,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    constraints: Optional[ConstraintSet] = None
) -> Tuple[Optional[PathVertex], float]:
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    constraints = constraints or ConstraintSet()
    
    def is_valid(node: Tuple[int, int], time: int, prev_node: Optional[Tuple[int, int]] = None) -> bool:
        vertex = PathVertex(node, time)
        if vertex in constraints:
            return False
        if prev_node is not None:
            edge = PathEdge(prev_node, node, time)
            if edge in constraints:
                return False
        return True

    open_set = [(0, start, 0, None)]
    came_from = {}
    g_score = {(start, 0): 0}
    f_score = {(start, 0): heuristic(start, goal)}
    goal_heuristic = {node: heuristic(node, goal) for node in env.graph.nodes()}

    while open_set:
        current_f, current_node, current_time, prev_node = heapq.heappop(open_set)

        if current_node == goal:
            path = []
            total_cost = g_score[(current_node, current_time)]
            while current_node:
                path.append(PathVertex(current_node, current_time))
                (current_node, current_time), prev_node = came_from.get(
                    (current_node, current_time), ((None, None), None)
                )
            path = path[::-1]  # Reverse path
            return Path(path, total_cost), total_cost 

        for neighbor in env.graph.neighbors(current_node):
            new_time = current_time + 1
            if not is_valid(neighbor, new_time, current_node):
                continue

            tentative_g_score = g_score[(current_node, current_time)] + 1
            if tentative_g_score < g_score.get((neighbor, new_time), float('inf')):
                came_from[(neighbor, new_time)] = ((current_node, current_time), prev_node)
                g_score[(neighbor, new_time)] = tentative_g_score
                f_score[(neighbor, new_time)] = tentative_g_score + goal_heuristic[neighbor]
                heapq.heappush(open_set, (f_score[(neighbor, new_time)], neighbor, new_time, current_node))

    return None, float('inf')