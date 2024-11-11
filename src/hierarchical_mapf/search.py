from typing import Dict, List, Tuple, Optional, Any
import heapq
import numpy as np

from ..common import PathVertex, PathEdge

def focal_search(
    action_gen: Any,
    V: Dict,
    E: Dict,
    start: PathVertex,
    goal: Any,
    constraints: Dict,
    omega: float
):
    open_set = []
    open_finder = {}
    focal_set = []
    focal_finder = {}
    predecessors = {}
    
    h = lambda loc: goal.heuristic(loc)
    
    if start in constraints:
        return None, np.inf, np.inf
    
    predecessors[start] = None
    d_score = 1 if start in V else 0
    g_score = 1
    f_score = g_score + h(start.pos)
    
    d = {start: d_score}
    g = {start: g_score}
    f = {start: f_score}

    heapq.heappush(open_set, [f_score, start])
    open_finder[start] = [f_score, start]
    f_best = f_score

    heapq.heappush(focal_set, [d_score, f_score, start])
    focal_finder[start] = [d_score, f_score, start]

    while focal_set or open_set:
        if open_set[0][0] != f_best:
            f_best = open_set[0][0]
            new_focal = []
            while focal_set:
                entry = heapq.heappop(focal_set)
                vertex = entry[2]
                focal_finder.pop(vertex)
                if f[vertex] <= omega * f_best:
                    focal_finder[vertex] = entry
                    heapq.heappush(new_focal, entry)
            focal_set = new_focal

        if focal_set:
            d_score, f_score, vertex = focal_set[0]
            if goal.satisfied(vertex.pos):
                return _reconstruct_path(predecessors, vertex)
            heapq.heappop(focal_set)
            focal_finder.pop(vertex)
        else:
            f_score, vertex = open_set[0]
            if goal.satisfied(vertex.pos):
                return _reconstruct_path(predecessors, vertex)
            heapq.heappop(open_set)
            open_finder.pop(vertex)

        successors = []
        for next_vertex, edge in action_gen.actions(vertex):
            if (next_vertex in constraints or 
                edge in constraints or 
                edge.compliment() in constraints):
                continue
            successors.append((next_vertex, edge))

        for next_vertex, edge in successors:
            tentative_g = g[vertex] + 1
            
            if next_vertex in g and tentative_g >= g[next_vertex]:
                continue
                
            predecessors[next_vertex] = vertex
            d_score = d[vertex]
            g_score = tentative_g
            f_score = g_score + h(next_vertex.pos)
            
            if vertex in V:
                d_score += 1
            if edge in E or edge.compliment() in E:
                d_score += 1
                
            d[next_vertex] = d_score
            g[next_vertex] = g_score
            f[next_vertex] = f_score

            if next_vertex not in open_finder:
                entry = [f_score, next_vertex]
                open_finder[next_vertex] = entry
                heapq.heappush(open_set, entry)
            else:
                entry = open_finder[next_vertex]
                if f_score != entry[0]:
                    entry[0] = f_score
                    heapq.heapify(open_set)
                    
            if f_score <= open_set[0][0] * omega:
                if next_vertex not in focal_finder:
                    entry = [d_score, f_score, next_vertex]
                    focal_finder[next_vertex] = entry
                    heapq.heappush(focal_set, entry)
                else:
                    entry = focal_finder[next_vertex]
                    if d_score != entry[0] or f_score != entry[1]:
                        entry[0] = d_score
                        entry[1] = f_score
                        heapq.heapify(focal_set)

    return None, np.inf, np.inf

def _reconstruct_path(predecessors: Dict, vertex: PathVertex):
    path = []
    current = vertex
    while current in predecessors:
        path.append(current)
        current = predecessors[current]
    path.append(current)
    path.reverse()
    return path, len(path), predecessors[vertex][0] if vertex in predecessors else np.inf

